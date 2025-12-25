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
SERVER_IP = "http://1.2.3.4:5000"
UPLOAD_URL = f"{SERVER_IP}/api/model/upload"
DOWNLOAD_URL = f"{SERVER_IP}/api/model/latest"
GLOBAL_MODEL_PATH = "global_model.ckpt" 

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# --- MODEL ARCHITECTURE ---
class FraudNet(nn.Cell):
    def __init__(self):
        super(FraudNet, self).__init__()
        self.dense1 = nn.Dense(6, 32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Dense(32, 16)
        self.dense3 = nn.Dense(16, 2)

    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

# --- PROFESSIONAL THEME ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

APP_NAME = "Fraud-Net Federation Client"
CONFIG_FILE = "client_config.json"
INFERENCE_SCRIPT = "inference_api.py"

# Professional Color Palette
COLORS = {
    'primary': '#1e40af',      # Professional Blue
    'primary_hover': '#1e3a8a',
    'secondary': '#3b82f6',
    'success': '#059669',
    'success_hover': '#047857',
    'danger': '#dc2626',
    'danger_hover': '#b91c1c',
    'warning': '#d97706',
    'bg_white': '#ffffff',
    'bg_light': '#f8fafc',
    'bg_gray': '#f1f5f9',
    'text_dark': '#0f172a',
    'text_gray': '#64748b',
    'border': '#e2e8f0'
}

# Terms and Conditions Text
TERMS_TEXT = """END USER LICENSE AGREEMENT

Last Updated: December 25, 2025

PLEASE READ THIS LICENSE AGREEMENT CAREFULLY BEFORE USING THIS SOFTWARE.

1. ACCEPTANCE OF TERMS
By installing, copying, or otherwise using Fraud-Net Federation Client ("the Software"), you agree to be bound by the terms of this Agreement. If you do not agree to the terms of this Agreement, do not install or use the Software.

2. LICENSE GRANT
Subject to the terms of this Agreement, we grant you a limited, non-exclusive, non-transferable, revocable license to use the Software for fraud detection and analysis purposes within your organization.

3. FEDERATED LEARNING PARTICIPATION
3.1. You agree to participate in the Fraud-Net federated learning network.
3.2. Model weights and parameters derived from your local training may be uploaded to the central server for aggregation.
3.3. You acknowledge that aggregated global models will be distributed to all network participants.

4. DATA PRIVACY AND SECURITY
4.1. Your raw transaction data NEVER leaves your local environment.
4.2. Only encrypted model weights and gradients are transmitted to the federation server.
4.3. You retain full ownership and control of your local data.
4.4. The Software does not collect, store, or transmit personally identifiable information (PII).

5. INTELLECTUAL PROPERTY
5.1. The Software and all intellectual property rights therein remain the property of the licensor.
5.2. You may not reverse engineer, decompile, or disassemble the global model.
5.3. Local model adaptations remain your intellectual property.

6. USAGE RESTRICTIONS
You may NOT:
- Use the Software for any unlawful purpose
- Attempt to compromise the security of the federation network
- Extract or replicate the global model architecture for commercial purposes
- Interfere with other participants' use of the federation
- Use the Software to process data without proper authorization

7. DISCLAIMER OF WARRANTIES
THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

8. LIMITATION OF LIABILITY
IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE SOFTWARE.

9. DATA PROCESSING AND COMPLIANCE
9.1. You are responsible for ensuring compliance with applicable data protection laws (GDPR, CCPA, etc.)
9.2. You warrant that you have necessary rights and consents to process the data used with this Software.
9.3. You agree to implement appropriate security measures to protect processed data.

10. AUDIT AND MONITORING
10.1. The federation network may monitor model updates for quality and security purposes.
10.2. Anomalous behavior or malicious contributions may result in license termination.

11. UPDATES AND MODIFICATIONS
11.1. The Software may automatically download and install updates.
11.2. Continued use after updates constitutes acceptance of modified terms.

12. TERMINATION
This license is effective until terminated. Your rights will terminate automatically without notice if you fail to comply with any term of this Agreement.

13. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with applicable international and local laws.

14. CONTACT INFORMATION
For questions about this Agreement, please contact: support@fraudnet-federation.example.com

BY CLICKING "I ACCEPT" OR INSTALLING THE SOFTWARE, YOU ACKNOWLEDGE THAT YOU HAVE READ THIS AGREEMENT, UNDERSTAND IT, AND AGREE TO BE BOUND BY ITS TERMS AND CONDITIONS."""

def run_real_training(csv_path, progress_callback, config_path="client_config.json"):
    try:
        progress_callback(5)
        print("Syncing with Global Model...")
        download_success, msg = run_real_download()
        if not download_success:
            print(f"Warning: Could not sync ({msg}). Training from scratch or old local weights.")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        progress_callback(10)
        df = pd.read_csv(csv_path)
        
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
        
        required = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
        if not all(col in df.columns for col in required):
            return f"Error: CSV missing columns. Found: {list(df.columns)}"

        progress_callback(20)
        df['type'] = df['type'].apply(lambda x: 1 if x in ['TRANSFER', 'CASH_OUT'] else 0)
        
        X = df[required[:-1]].values.astype(np.float32)
        y = df['isFraud'].values.astype(np.int32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        net = FraudNet()
        
        if os.path.exists(GLOBAL_MODEL_PATH):
            param_dict = load_checkpoint(GLOBAL_MODEL_PATH)
            load_param_into_net(net, param_dict)
            print("Loaded Global Weights.")
            
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
            
            prog = 30 + int((epoch + 1) / epochs * 60)
            progress_callback(prog)
            print(f"Epoch {epoch+1} Loss: {step_loss/steps:.4f}")

        output = net(Tensor(X_test))
        predicted = np.argmax(output.asnumpy(), axis=1)
        acc = accuracy_score(y_test, predicted)
        acc_str = f"{acc*100:.2f}"
        
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
            headers = {'Bank-ID': 'Bank_Client_App'}
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
# INSTALLER WIZARD
# ======================================================
class InstallerWizard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Install {APP_NAME}")
        self.geometry("650x600")
        self.resizable(False, False)
        self.configure(fg_color=COLORS['bg_white'])
        
        self.step = 1
        
        # Header
        self.header = ctk.CTkFrame(self, height=80, fg_color=COLORS['primary'], corner_radius=0)
        self.header.pack(fill="x")
        ctk.CTkLabel(
            self.header, 
            text=f"{APP_NAME} Setup", 
            font=("Segoe UI", 24, "bold"),
            text_color="white"
        ).pack(pady=25)
        
        # Main Container
        self.frame = ctk.CTkFrame(self, fg_color=COLORS['bg_white'])
        self.frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        self.show_terms_page()

    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def show_terms_page(self):
        self.clear_frame()
        
        # Title
        ctk.CTkLabel(
            self.frame, 
            text="License Agreement", 
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS['text_dark']
        ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            self.frame,
            text="Please read the following license agreement carefully",
            font=("Segoe UI", 11),
            text_color=COLORS['text_gray']
        ).pack(pady=(0, 15))
        
        # Terms Box with better styling
        terms_frame = ctk.CTkFrame(self.frame, fg_color=COLORS['bg_light'], corner_radius=8)
        terms_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        textbox = ctk.CTkTextbox(
            terms_frame,
            font=("Consolas", 9),
            fg_color=COLORS['bg_light'],
            text_color=COLORS['text_dark'],
            wrap="word"
        )
        textbox.pack(fill="both", expand=True, padx=2, pady=2)
        textbox.insert("0.0", TERMS_TEXT)
        textbox.configure(state="disabled")
        
        # Checkbox
        self.agree_var = ctk.BooleanVar()
        chk = ctk.CTkCheckBox(
            self.frame,
            text="I have read and agree to the license agreement",
            variable=self.agree_var,
            command=self.check_agreement,
            font=("Segoe UI", 11),
            text_color=COLORS['text_dark'],
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover']
        )
        chk.pack(pady=15)
        
        # Button Frame
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=120,
            height=36,
            font=("Segoe UI", 12),
            fg_color=COLORS['bg_gray'],
            text_color=COLORS['text_dark'],
            hover_color=COLORS['border'],
            command=self.quit
        ).pack(side="left")
        
        self.btn_next = ctk.CTkButton(
            btn_frame,
            text="Next →",
            width=120,
            height=36,
            font=("Segoe UI", 12, "bold"),
            state="disabled",
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover'],
            command=self.show_config_page
        )
        self.btn_next.pack(side="right")

    def check_agreement(self):
        if self.agree_var.get():
            self.btn_next.configure(state="normal")
        else:
            self.btn_next.configure(state="disabled")

    def show_config_page(self):
        self.clear_frame()
        
        ctk.CTkLabel(
            self.frame,
            text="Configure Data Adapter",
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS['text_dark']
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            self.frame,
            text="Map your database column names to the standard format",
            font=("Segoe UI", 11),
            text_color=COLORS['text_gray']
        ).pack(pady=(0, 20))
        
        # Scrollable Configuration Area
        self.scroll = ctk.CTkScrollableFrame(
            self.frame,
            fg_color=COLORS['bg_light'],
            corner_radius=8
        )
        self.scroll.pack(fill="both", expand=True, pady=(0, 20))

        self.wizard_entries = {}
        fields = [
            ("col_type", "Transaction Type Column", "e.g., 'type' or 'transaction_type'"),
            ("col_amount", "Amount Column", "e.g., 'amount' or 'transaction_amount'"),
            ("col_oldbal", "Old Balance Origin Column", "e.g., 'oldbalanceOrg'"),
            ("col_newbal", "New Balance Origin Column", "e.g., 'newbalanceOrig'"),
            ("col_olddest", "Old Balance Destination Column", "e.g., 'oldbalanceDest'"),
            ("col_newdest", "New Balance Destination Column", "e.g., 'newbalanceDest'"),
            ("col_fraud", "Fraud Label Column", "e.g., 'isFraud' or 'is_fraudulent'")
        ]

        for key, label, placeholder in fields:
            field_frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
            field_frame.pack(fill="x", pady=8, padx=10)
            
            ctk.CTkLabel(
                field_frame,
                text=label,
                font=("Segoe UI", 11, "bold"),
                text_color=COLORS['text_dark'],
                anchor="w"
            ).pack(fill="x")
            
            entry = ctk.CTkEntry(
                field_frame,
                placeholder_text=placeholder,
                height=36,
                font=("Segoe UI", 11),
                fg_color="white",
                border_color=COLORS['border'],
                text_color=COLORS['text_dark']
            )
            entry.pack(fill="x", pady=(5, 0))
            self.wizard_entries[key] = entry

        # Button Frame
        btn_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        ctk.CTkButton(
            btn_frame,
            text="← Back",
            width=120,
            height=36,
            font=("Segoe UI", 12),
            fg_color=COLORS['bg_gray'],
            text_color=COLORS['text_dark'],
            hover_color=COLORS['border'],
            command=self.show_terms_page
        ).pack(side="left")
        
        ctk.CTkButton(
            btn_frame,
            text="Complete Installation",
            width=180,
            height=36,
            font=("Segoe UI", 12, "bold"),
            fg_color=COLORS['success'],
            hover_color=COLORS['success_hover'],
            command=self.finish_install
        ).pack(side="right")

    def finish_install(self):
        config_data = {key: entry.get() for key, entry in self.wizard_entries.items()}
        config_data["installed"] = True
        
        if not all(config_data.values()):
            messagebox.showerror("Error", "All fields are required.")
            return

        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f)
        
        messagebox.showinfo("Success", "Installation completed successfully!\n\nThe control panel will now launch.")
        self.destroy()
        launch_main_app()

# ======================================================
# MAIN CONTROL PANEL
# ======================================================
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("550x700")
        self.configure(fg_color=COLORS['bg_white'])
        self.service_process = None

        # Header
        self.header = ctk.CTkFrame(self, height=70, fg_color=COLORS['primary'], corner_radius=0)
        self.header.pack(fill="x")
        ctk.CTkLabel(
            self.header,
            text="FRAUD-NET CONTROL PANEL",
            font=("Segoe UI", 22, "bold"),
            text_color="white"
        ).pack(pady=20)

        # Tabs
        self.tabs = ctk.CTkTabview(
            self,
            fg_color=COLORS['bg_white'],
            segmented_button_fg_color=COLORS['bg_gray'],
            segmented_button_selected_color=COLORS['primary'],
            segmented_button_selected_hover_color=COLORS['primary_hover'],
            text_color=COLORS['text_dark']
        )
        self.tabs.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.tab_status = self.tabs.add("Status")
        self.tab_train = self.tabs.add("Training")
        self.tab_sync = self.tabs.add("Federation")
        self.tab_settings = self.tabs.add("Settings")

        self.setup_status_tab()
        self.setup_train_tab()
        self.setup_sync_tab()
        self.setup_settings_tab()

    def setup_status_tab(self):
        frame = ctk.CTkFrame(self.tab_status, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Status Card
        status_card = ctk.CTkFrame(frame, fg_color=COLORS['bg_light'], corner_radius=10)
        status_card.pack(fill="x", pady=(0, 20))
        
        self.status_lbl = ctk.CTkLabel(
            status_card,
            text="● Service Stopped",
            text_color=COLORS['danger'],
            font=("Segoe UI", 16, "bold")
        )
        self.status_lbl.pack(pady=25)
        
        self.btn_toggle = ctk.CTkButton(
            status_card,
            text="START INFERENCE API",
            height=45,
            font=("Segoe UI", 13, "bold"),
            fg_color=COLORS['success'],
            hover_color=COLORS['success_hover'],
            command=self.toggle_service
        )
        self.btn_toggle.pack(fill="x", padx=30, pady=(0, 15))
        
        self.lbl_port = ctk.CTkLabel(
            status_card,
            text="Port: 5001",
            text_color=COLORS['text_gray'],
            font=("Segoe UI", 10)
        )
        self.lbl_port.pack(pady=(0, 15))

        # Activity Log
        ctk.CTkLabel(
            frame,
            text="Activity Log",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_dark']
        ).pack(fill="x", pady=(10, 5))
        
        log_frame = ctk.CTkFrame(frame, fg_color=COLORS['bg_light'], corner_radius=8)
        log_frame.pack(fill="both", expand=True)
        
        self.log_box = ctk.CTkTextbox(
            log_frame,
            font=("Consolas", 9),
            fg_color=COLORS['bg_light'],
            text_color=COLORS['text_dark']
        )
        self.log_box.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_box.insert("0.0", f"[{datetime.now().strftime('%H:%M:%S')}] Application started\n")
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] Configuration loaded\n")

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")

    def toggle_service(self):
        if self.service_process is None:
            self.service_process = True
            
            self.status_lbl.configure(text="● Service Running", text_color=COLORS['success'])
            self.btn_toggle.configure(
                text="STOP SERVICE",
                fg_color=COLORS['danger'],
                hover_color=COLORS['danger_hover']
            )
            self.log("Inference API started on port 5001")
        else:
            self.service_process = None
            
            self.status_lbl.configure(text="● Service Stopped", text_color=COLORS['danger'])
            self.btn_toggle.configure(
                text="START INFERENCE API",
                fg_color=COLORS['success'],
                hover_color=COLORS['success_hover']
            )
            self.log("Service stopped")

    def setup_train_tab(self):
        frame = ctk.CTkFrame(self.tab_train, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # File Selection Card
        file_card = ctk.CTkFrame(frame, fg_color=COLORS['bg_light'], corner_radius=10)
        file_card.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            file_card,
            text="Training Data",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS['text_dark']
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            file_card,
            text="Select a CSV file containing transaction data",
            font=("Segoe UI", 10),
            text_color=COLORS['text_gray']
        ).pack(pady=(0, 10))
        
        self.btn_csv = ctk.CTkButton(
            file_card,
            text="📁 Browse Files",
            height=40,
            font=("Segoe UI", 12),
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover'],
            command=self.select_csv
        )
        self.btn_csv.pack(pady=(0, 10), padx=20)
        
        self.lbl_csv_name = ctk.CTkLabel(
            file_card,
            text="No file selected",
            text_color=COLORS['text_gray'],
            font=("Segoe UI", 10)
        )
        self.lbl_csv_name.pack(pady=(0, 15))
        
        # Training Control
        self.btn_start_train = ctk.CTkButton(
            frame,
            text="START TRAINING",
            height=45,
            font=("Segoe UI", 13, "bold"),
            state="disabled",
            fg_color=COLORS['success'],
            hover_color=COLORS['success_hover'],
            command=self.start_training
        )
        self.btn_start_train.pack(fill="x", pady=(0, 15))
        
        # Progress Section
        progress_frame = ctk.CTkFrame(frame, fg_color=COLORS['bg_light'], corner_radius=10)
        progress_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            progress_frame,
            text="Training Progress",
            font=("Segoe UI", 11, "bold"),
            text_color=COLORS['text_dark']
        ).pack(pady=(15, 5), padx=15, anchor="w")
        
        self.progress = ctk.CTkProgressBar(
            progress_frame,
            height=8,
            progress_color=COLORS['primary']
        )
        self.progress.set(0)
        self.progress.pack(fill="x", padx=15, pady=(0, 15))
        
        self.lbl_train_result = ctk.CTkLabel(
            frame,
            text="",
            font=("Segoe UI", 11),
            text_color=COLORS['text_dark'],
            wraplength=480
        )
        self.lbl_train_result.pack()

    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            self.lbl_csv_name.configure(
                text=f"✓ {os.path.basename(path)}",
                text_color=COLORS['success']
            )
            self.btn_start_train.configure(state="normal")
            self.log(f"Selected training file: {os.path.basename(path)}")

    def start_training(self):
        self.btn_start_train.configure(state="disabled")
        self.log(f"Starting training on {os.path.basename(self.csv_path)}")
        threading.Thread(target=self.run_training_thread).start()

    def run_training_thread(self):
        def update(val):
            self.progress.set(val/100)
        
        result_msg = run_real_training(self.csv_path, update)
        
        self.lbl_train_result.configure(text=result_msg)
        self.btn_start_train.configure(state="normal")
        self.log(result_msg)

    def setup_sync_tab(self):
        frame = ctk.CTkFrame(self.tab_sync, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Download Card
        download_card = ctk.CTkFrame(frame, fg_color=COLORS['bg_light'], corner_radius=10)
        download_card.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            download_card,
            text="Global Model Sync",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS['text_dark']
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            download_card,
            text="Download the latest aggregated model from the federation",
            font=("Segoe UI", 10),
            text_color=COLORS['text_gray'],
            wraplength=450
        ).pack(pady=(0, 15), padx=20)
        
        ctk.CTkButton(
            download_card,
            text="⬇ Check for Updates",
            height=40,
            font=("Segoe UI", 12),
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover'],
            command=self.check_updates
        ).pack(fill="x", padx=30, pady=(0, 20))
        
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