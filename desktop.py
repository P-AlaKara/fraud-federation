import customtkinter as ctk
import os
import json
import threading
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import nn, context, save_checkpoint, load_checkpoint, load_param_into_net, Tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import filedialog, messagebox

# --- CONFIGURATION ---
SERVER_IP = "http://1.2.3.4:5000"  # <--- REPLACE WITH YOUR AWS IP
UPLOAD_URL = f"{SERVER_IP}/api/model/upload"
DOWNLOAD_URL = f"{SERVER_IP}/api/model/latest"
GLOBAL_MODEL_PATH = "global_model.ckpt" 

# Ensure CPU usage (Client side)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# --- THE MODEL ARCHITECTURE (MUST MATCH SERVER) ---
class FraudNet(nn.Cell):
    def __init__(self):
        super(FraudNet, self).__init__()
        # 6 Inputs: type, amount, oldBalOrg, newBalOrig, oldBalDest, newBalDest
        self.dense1 = nn.Dense(6, 32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Dense(32, 16)
        self.dense3 = nn.Dense(16, 2) # Output: 2 classes (Fraud/Legit)

    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

# --- CONFIGURATION ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

APP_NAME = "Fraud-Net Federation Client"
CONFIG_FILE = "client_config.json"
INFERENCE_SCRIPT = "inference_api.py" # The actual backend script file

def run_real_training(csv_path, progress_callback, config_path="client_config.json"):
    try:
        progress_callback(5)
        print("Syncing with Global Model...")
        download_success, msg = run_real_download()
        if not download_success:
            print(f"Warning: Could not sync ({msg}). Training from scratch or old local weights.")
        
        # 1. LOAD CONFIG
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # 2. LOAD & ADAPT DATA
        progress_callback(10)
        df = pd.read_csv(csv_path)
        
        # Dynamic Renaming from Config
        rename_map = {
            config.get("col_type", "type"): "type",
            config.get("col_amount", "amount"): "amount",
            config.get("col_oldbal", "oldbalanceOrg"): "oldbalanceOrg",
            config.get("col_newbal", "newbalanceOrig"): "newbalanceOrig",
            config.get("col_olddest", "oldbalanceDest"): "oldbalanceDest",
            config.get("col_newdest", "newbalanceDest"): "newbalanceDest",
            config.get("col_fraud", "isFraud"): "isFraud"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Validation
        required = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
        if not all(col in df.columns for col in required):
            return f"Error: CSV missing columns. Found: {list(df.columns)}"

        # 3. PREPROCESSING
        progress_callback(20)
        df['type'] = df['type'].apply(lambda x: 1 if x in ['TRANSFER', 'CASH_OUT'] else 0)
        
        X = df[required[:-1]].values.astype(np.float32)
        y = df['isFraud'].values.astype(np.int32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. INITIALIZE MODEL
        net = FraudNet()
        
        # Load Global Weights (which we just downloaded in Step 0)
        if os.path.exists(GLOBAL_MODEL_PATH):
            param_dict = load_checkpoint(GLOBAL_MODEL_PATH)
            load_param_into_net(net, param_dict)
            print("Loaded Global Weights.")
            
        # 5. TRAINING LOOP
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
        train_net = nn.TrainOneStepCell(nn.WithLossCell(net, loss_fn), optimizer)
        train_net.set_train()

        epochs = 5
        batch_size = 32
        steps = X_train.shape[0] // batch_size
        
        for epoch in range(epochs):
            step_loss = 0
            for i in range(0, X_train.shape[0], batch_size):
                end = i + batch_size
                batch_X = Tensor(X_train[i:end])
                batch_y = Tensor(y_train[i:end])
                loss = train_net(batch_X, batch_y)
                step_loss += float(loss.asnumpy())
            
            # Progress: 30% to 90%
            prog = 30 + int((epoch + 1) / epochs * 60)
            progress_callback(prog)
            print(f"Epoch {epoch+1} Loss: {step_loss/steps:.4f}")

        # 6. SAVE WITH VERSIONING
        output = net(Tensor(X_test))
        predicted = np.argmax(output.asnumpy(), axis=1)
        acc = accuracy_score(y_test, predicted)
        acc_str = f"{acc*100:.2f}"
        
        # Versioning using Unix Timestamp
        version_id = int(time.time())
        save_name = f"local_v{version_id}_acc{acc_str}.ckpt"
        
        save_checkpoint(net, save_name)
        
        progress_callback(100)
        return f"Training Done! Accuracy: {acc_str}% | Saved: {save_name}"

    except Exception as e:
        print(e)
        return f"Training Failed: {str(e)}"

def run_real_upload(file_path):
    try:
        if not os.path.exists(file_path):
            return False, "File not found."
            
        with open(file_path, 'rb') as f:
            files = {'file': f}
            headers = {'Bank-ID': 'Bank_Client_App'} # TODO: use api keysIdeally configurable
            r = requests.post(UPLOAD_URL, files=files, headers=headers)
            
        if r.status_code == 200:
            return True, r.json().get('message', 'Upload OK')
        else:
            return False, f"Server Error: {r.status_code}"
    except Exception as e:
        return False, str(e)

def run_real_download():
    try:
        r = requests.get(DOWNLOAD_URL)
        if r.status_code == 200:
            with open(GLOBAL_MODEL_PATH, 'wb') as f:
                f.write(r.content)
            return True, "Global Model Updated."
        elif r.status_code == 404:
            return False, "No Global Model exists on server yet."
        else:
            return False, f"Error: {r.status_code}"
    except Exception as e:
        return False, str(e)

# ======================================================
# SCREEN 1: THE INSTALLATION WIZARD
# ======================================================
class InstallerWizard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Install {APP_NAME}")
        self.geometry("500x450")
        self.resizable(False, False)
        
        self.step = 1
        
        # Main Container
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.show_terms_page()

    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def show_terms_page(self):
        self.clear_frame()
        ctk.CTkLabel(self.frame, text="Step 1: Agreement", font=("Arial", 20, "bold")).pack(pady=10)
        
        terms = "1. You agree to participate in the Federation.\n2. You will not attempt to reverse engineer the Global Model.\n3. Local data remains on your machine.\n4. Only Model Weights are uploaded."
        
        textbox = ctk.CTkTextbox(self.frame, height=150)
        textbox.insert("0.0", terms)
        textbox.configure(state="disabled")
        textbox.pack(fill="x", pady=10)
        
        self.agree_var = ctk.BooleanVar()
        chk = ctk.CTkCheckBox(self.frame, text="I accept the terms", variable=self.agree_var, command=self.check_agreement)
        chk.pack(pady=10)
        
        self.btn_next = ctk.CTkButton(self.frame, text="Next >", state="disabled", command=self.show_config_page)
        self.btn_next.pack(side="bottom", pady=10)

    def check_agreement(self):
        if self.agree_var.get():
            self.btn_next.configure(state="normal")
        else:
            self.btn_next.configure(state="disabled")

    def show_config_page(self):
        self.clear_frame()
        ctk.CTkLabel(self.frame, text="Step 2: Adapter Config", font=("Arial", 20, "bold")).pack(pady=10)
        ctk.CTkLabel(self.frame, text="Map your database columns to our standard.", text_color="gray").pack()
        
        # Create a scrollable area for the 7 fields
        self.scroll = ctk.CTkScrollableFrame(self.frame, height=250)
        self.scroll.pack(fill="both", expand=True, pady=10)

        # The 7 Required Fields
        self.wizard_entries = {}
        fields = [
            ("col_type", "Transaction Type (e.g. type)"),
            ("col_amount", "Amount (e.g. amount)"),
            ("col_oldbal", "Old Balance Origin (e.g. oldbalanceOrg)"),
            ("col_newbal", "New Balance Origin (e.g. newbalanceOrig)"),
            ("col_olddest", "Old Balance Dest (e.g. oldbalanceDest)"),
            ("col_newdest", "New Balance Dest (e.g. newbalanceDest)"),
            ("col_fraud", "Target Label (e.g. isFraud)")
        ]

        # Loop to create inputs
        for key, label in fields:
            ctk.CTkLabel(self.scroll, text=label, anchor="w").pack(fill="x")
            entry = ctk.CTkEntry(self.scroll)
            entry.pack(fill="x", pady=(0, 10))
            self.wizard_entries[key] = entry

        ctk.CTkButton(self.frame, text="Finish Installation", fg_color="green", command=self.finish_install).pack(side="bottom", pady=20)

    def finish_install(self):
        # Harvest data from the loop
        config_data = {key: entry.get() for key, entry in self.wizard_entries.items()}
        config_data["installed"] = True
        
        # Basic validation
        if not all(config_data.values()):
            messagebox.showerror("Error", "All fields are required.")
            return

        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f)
        
        messagebox.showinfo("Success", "Installation Complete!\nLaunching Control Panel...")
        self.destroy()
        launch_main_app()

# ======================================================
# SCREEN 2: THE MAIN CONTROL PANEL (With Settings Tab)
# ======================================================
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("450x650")
        self.service_process = None

        # --- HEADER ---
        self.header = ctk.CTkFrame(self, height=60, fg_color="#1f538d")
        self.header.pack(fill="x")
        ctk.CTkLabel(self.header, text="FRAUD-NET HUB", font=("Arial", 22, "bold"), text_color="white").pack(pady=15)

        # --- TABS ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_status = self.tabs.add("Status")
        self.tab_train = self.tabs.add("Train")
        self.tab_sync = self.tabs.add("Sync")
        self.tab_settings = self.tabs.add("Settings") # <--- NEW TAB

        self.setup_status_tab()
        self.setup_train_tab()
        self.setup_sync_tab()
        self.setup_settings_tab()

    # --- TAB 1: STATUS ---
    def setup_status_tab(self):
        frame = self.tab_status
        
        self.status_lbl = ctk.CTkLabel(frame, text="● SERVICE STOPPED", text_color="red", font=("Arial", 14, "bold"))
        self.status_lbl.pack(pady=30)
        
        self.btn_toggle = ctk.CTkButton(frame, text="START INFERENCE API", height=50, font=("Arial", 14), 
                                        fg_color="green", hover_color="darkgreen", command=self.toggle_service)
        self.btn_toggle.pack(fill="x", padx=40, pady=10)
        
        self.lbl_port = ctk.CTkLabel(frame, text="Port: 5001", text_color="gray")
        self.lbl_port.pack()

        # Logs Console
        ctk.CTkLabel(frame, text="Activity Log:", anchor="w").pack(fill="x", padx=10, pady=(20,0))
        self.log_box = ctk.CTkTextbox(frame, height=150)
        self.log_box.insert("0.0", f"> App Started.\n> Config loaded from {CONFIG_FILE}\n")
        self.log_box.pack(fill="both", padx=10, pady=5)

    def log(self, message):
        self.log_box.insert("end", f"> {message}\n")
        self.log_box.see("end")

    def toggle_service(self):
        if self.service_process is None:
            # TODO: REAL LOGIC -> Use subprocess.Popen(['python', INFERENCE_SCRIPT])
            self.service_process = True 
            
            self.status_lbl.configure(text="● SERVICE RUNNING", text_color="#00ff00")
            self.btn_toggle.configure(text="STOP SERVICE", fg_color="red", hover_color="darkred")
            self.log("Inference API started on port 5001.")
        else:
            # TODO: REAL LOGIC -> self.service_process.terminate()
            self.service_process = None
            
            self.status_lbl.configure(text="● SERVICE STOPPED", text_color="red")
            self.btn_toggle.configure(text="START INFERENCE API", fg_color="green", hover_color="darkgreen")
            self.log("Service stopped.")

    # --- TAB 2: TRAIN ---
    def setup_train_tab(self):
        frame = self.tab_train
        
        ctk.CTkLabel(frame, text="Select Training Data (CSV)", font=("Arial", 14)).pack(pady=15)
        
        self.btn_csv = ctk.CTkButton(frame, text="Browse Files...", command=self.select_csv)
        self.btn_csv.pack(pady=5)
        
        self.lbl_csv_name = ctk.CTkLabel(frame, text="No file selected", text_color="gray")
        self.lbl_csv_name.pack(pady=5)
        
        ctk.CTkFrame(frame, height=2, fg_color="gray").pack(fill="x", padx=20, pady=20)
        
        self.btn_start_train = ctk.CTkButton(frame, text="START TRAINING", state="disabled", command=self.start_training)
        self.btn_start_train.pack(fill="x", padx=40)
        
        self.progress = ctk.CTkProgressBar(frame)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=40, pady=20)
        
        self.lbl_train_result = ctk.CTkLabel(frame, text="")
        self.lbl_train_result.pack()

    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.csv_path = path
            self.lbl_csv_name.configure(text=os.path.basename(path), text_color="white")
            self.btn_start_train.configure(state="normal")

    def start_training(self):
        self.btn_start_train.configure(state="disabled")
        self.log(f"Starting Training on {os.path.basename(self.csv_path)}...")
        threading.Thread(target=self.run_training_thread).start()

    def run_training_thread(self):
        def update(val):
            self.progress.set(val/100)
        
        # CALL REAL BACKEND HERE
        result_msg = run_real_training(self.csv_path, update)
        
        # Update UI needs to be scheduled back on main thread in standard Tkinter, 
        # but CTk usually handles thread safety reasonably well for simple label updates.
        self.lbl_train_result.configure(text=result_msg)
        self.btn_start_train.configure(state="normal")
        self.log(result_msg)

    # --- TAB 3: SYNC ---
    def setup_sync_tab(self):
        frame = self.tab_sync
        
        # Download Section
        ctk.CTkLabel(frame, text="Global Model", font=("Arial", 14, "bold")).pack(pady=(20,5))
        ctk.CTkButton(frame, text="Check for Updates (Downlink)", command=self.check_updates).pack(fill="x", padx=40, pady=5)
        
        ctk.CTkFrame(frame, height=2, fg_color="gray").pack(fill="x", padx=20, pady=20)
        
        # Upload Section
        ctk.CTkLabel(frame, text="Local Contribution", font=("Arial", 14, "bold")).pack(pady=5)
        ctk.CTkLabel(frame, text="Share your learned patterns (weights only).").pack(pady=5)
        ctk.CTkButton(frame, text="Upload Latest Weights (Uplink)", fg_color="#D4AF37", text_color="black", hover_color="#B5952F", command=self.upload_weights).pack(fill="x", padx=40, pady=10)

    def check_updates(self):
        self.log("Connecting to Federation Server...")
        success, msg = run_real_download()
        if success:
            messagebox.showinfo("Sync", msg)
            self.log(f"Success: {msg}")
        else:
            messagebox.showerror("Sync Error", msg)
            self.log(f"Failed: {msg}")

    def upload_weights(self):
        # 1. Find the best local model
        files = [f for f in os.listdir('.') if f.startswith("local_model_acc") and f.endswith(".ckpt")]
        if not files:
            messagebox.showwarning("Upload", "No trained local models found.")
            return

        # Pick the latest one (or sort by accuracy if you prefer)
        latest_model = max(files, key=os.path.getctime) # Sort by creation time
        
        if messagebox.askyesno("Confirm", f"Upload {latest_model}?"):
            self.log(f"Uploading {latest_model}...")
            
            success, msg = run_real_upload(latest_model)
            
            if success:
                messagebox.showinfo("Success", msg)
                self.log(f"Upload Result: {msg}")
            else:
                messagebox.showerror("Error", msg)
                self.log(f"Upload Failed: {msg}")

    # --- TAB 4: SETTINGS (EDIT CONFIG) ---
    def setup_settings_tab(self):
        frame = self.tab_settings
        
        ctk.CTkLabel(frame, text="Adapter Configuration", font=("Arial", 16, "bold")).pack(pady=20)
        ctk.CTkLabel(frame, text="Modify these if your database schema changes.", text_color="gray").pack(pady=(0, 20))
        
        # Load current config
        current = {}
        if os.path.exists("client_config.json"):
            with open("client_config.json", "r") as f:
                current = json.load(f)
        
        # Create Scrollable Frame if many inputs, or just pack tightly
        self.scroll = ctk.CTkScrollableFrame(frame, label_text="Column Mapping")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Helper to create inputs
        self.entries = {}
        fields = [
            ("col_type", "Transaction Type (e.g. type)"),
            ("col_amount", "Amount (e.g. amount)"),
            ("col_oldbal", "Old Balance Origin (e.g. oldbalanceOrg)"),
            ("col_newbal", "New Balance Origin (e.g. newbalanceOrig)"),
            ("col_olddest", "Old Balance Dest (e.g. oldbalanceDest)"),
            ("col_newdest", "New Balance Dest (e.g. newbalanceDest)"),
            ("col_fraud", "Target Label (e.g. isFraud)")
        ]

        for key, label in fields:
            ctk.CTkLabel(self.scroll, text=label, anchor="w").pack(fill="x")
            entry = ctk.CTkEntry(self.scroll)
            entry.pack(fill="x", pady=(0, 10))
            entry.insert(0, current.get(key, ""))
            self.entries[key] = entry
        
        ctk.CTkButton(frame, text="Save Config", fg_color="green", command=self.save_settings).pack(pady=10)

    def save_settings(self):
        new_conf = {key: entry.get() for key, entry in self.entries.items()}
        new_conf["installed"] = True
        
        with open("client_config.json", "w") as f:
            json.dump(new_conf, f)
        
        messagebox.showinfo("Saved", "Adapter Configuration updated.")

    def save_settings(self):
        new_conf = {
            "amount_col": self.edit_amt.get(),
            "time_col": self.edit_time.get(),
            "installed": True
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_conf, f)
        
        messagebox.showinfo("Saved", "Configuration updated successfully.")
        self.log("Configuration updated by user.")

# --- ENTRY POINT ---
def launch_main_app():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        # Case 1: First time run -> Wizard
        app = InstallerWizard()
        app.mainloop()
    else:
        # Case 2: Already configured -> Control Panel
        launch_main_app()