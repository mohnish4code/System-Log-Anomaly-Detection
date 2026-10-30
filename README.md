# 🖥️ System Log Anomaly Detection

Automatically detecting **abnormal behaviour in large-scale computer systems from their log files**, so engineers are alerted to failures before they escalate into outages.

---

## 🚧 Status

**In progress.**

| Phase | Scope | State |
|---|---|---|
| 1 | Download, Drain parsing, grouping, chronological split, count vectors, rule / PCA / Isolation Forest / supervised baselines, metrics | ✅ Done |
| 2 | LSTM (DeepLog-style) and Transformer sequence models | ⏳ Next |
| 3 | Streaming replay, detection delay, FastAPI `/score`, Streamlit dashboard | ⏳ Planned |
| 4 | MLflow tracking, Docker, GitHub Actions | ⏳ Planned |

---

## 📌 Problem

Modern software systems (cloud platforms, distributed storage, supercomputers, microservices) write millions of log lines every day. These logs record what the system is doing, and failures usually leave traces in them before users notice a problem.

However:

- **The volume is far too large** for engineers to read manually.
- **Keyword rules** such as searching for `ERROR` miss many real problems and trigger on harmless messages.
- **Many failures are visible only as unusual *sequences*** of otherwise normal-looking messages, such as a step that is skipped or repeated.

This project builds a pipeline that converts raw, unstructured log text into structured events and uses machine learning to flag sessions or time windows whose behaviour differs from normal operation.

```
Raw log lines  →  Log templates  →  Event sequences  →  Anomaly score  →  Alert
```

---

## 💼 Why It Matters

- **Faster incident detection:** problems are caught earlier, reducing downtime and its cost.
- **Less manual effort:** engineers investigate flagged windows instead of searching through millions of lines.
- **Detects unknown failures:** unsupervised methods can flag problems that no one has written a rule for.
- **Supports reliability goals:** Site Reliability Engineering (SRE) and operations teams rely on early detection to meet uptime commitments.

Log anomaly detection is a core part of **AIOps** (AI for IT Operations) and is used by cloud providers, observability platforms, banks, telecom operators, and large software companies.

---

## 📊 Datasets

All datasets come from **[Loghub](https://github.com/logpai/loghub)**, a public collection of real system logs maintained by the LogPAI research group for AI-driven log analytics research.

### HDFS (Hadoop Distributed File System)
- **~11 million log lines** from a Hadoop cluster
- Logs are grouped by **block ID** into sessions (about 575,000 sessions)
- **Labels per session:** normal or anomalous (roughly 3% anomalous)
- Ideal for **session-based** anomaly detection

### BGL (Blue Gene/L Supercomputer)
- **~4.7 million log lines** from the Blue Gene/L supercomputer at Lawrence Livermore National Laboratory
- **Labels per line:** alert or non-alert
- Logs are grouped into **time windows**, since there is no natural session ID
- Harder than HDFS because it contains many rare and previously unseen message types

### Data Challenges
- **Unstructured text:** each line mixes a fixed message with variable values (IPs, block IDs, file paths, numbers), which must be separated before modeling.
- **Severe class imbalance:** anomalies are a small fraction of the data.
- **Unseen log messages:** new message types appear in the test period that never occurred in training.
- **Leakage risk:** randomly shuffling sessions into train and test leaks future information and inflates scores, so a chronological split is required.
- **Scale:** millions of lines require efficient, streaming-friendly processing.

---

## 🔧 Approach

```
Raw Logs (.log)
   ↓
Log Parsing (Drain)     →  extract templates, e.g. "Receiving block <BLK> src: /<IP> dest: /<IP>"
   ↓
Grouping                →  sessions (HDFS block ID) or sliding time windows (BGL)
   ↓
Feature Representation  →  event count vectors / ordered event sequences
   ↓
Chronological Split     →  train on earlier data, test on later data
   ↓
Models                  →  statistical → ML → deep sequence models
   ↓
Evaluation              →  precision, recall, F1, false alarms, detection delay
   ↓
Deployment              →  streaming simulation + FastAPI + Streamlit dashboard
```

### 1. Log Parsing
Raw log line:
```
081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_38865049064139660 of size 67108864 from /10.251.42.84
```
Parsed template:
```
Received block <BLK> of size <NUM> from /<IP>
```
Parsing uses **Drain**, a fixed-depth parse-tree algorithm that is fast and works in an online (streaming) setting through the `drain3` library. Before Drain sees a line, regex masks replace block IDs, IPs, hex values and numbers. Each unique template becomes an **event ID**.

The raw file is parsed once, in its original order, exactly as an online parser would see it. Templates that first appear in the test period are therefore never available at training time.

**Parsing accuracy** (grouping accuracy on Loghub's hand-labelled 2k-line samples, `sim_th = 0.4`):

| Dataset | Templates found / true | Grouping accuracy |
|---|---|---|
| HDFS | 16 / 14 | 0.998 |
| BGL | 105 / 120 | 0.969 |

### 2. Grouping
- **HDFS:** all events for the same block ID form one session. A line that mentions several blocks belongs to each of them.
- **BGL:** events are grouped into fixed time windows (default 60 min; sliding windows via `--step-minutes`). A window is anomalous if any of its lines is an alert.

### 3. Feature Representation
| Representation | Used by | What it captures |
|---|---|---|
| **Event count vector** | PCA, Isolation Forest, supervised baselines | How often each event type occurs in a session |
| **Ordered event sequence** | LSTM / Transformer models | The order in which events occur |
| **Time between events** | Extension | Unusual delays or bursts |

The count-vector vocabulary is learned from the training split only. Event types first seen later go into a single `<UNSEEN>` column, so a burst of new templates is itself a signal.

### 4. Chronological Split
Sessions are sorted by start time and split 60 / 20 / 20 into train / validation / test. For sliding windows, `--purge-overlap` drops windows that share log lines with the previous split.

---

## 🤖 Models

| Model | Type | How it detects anomalies |
|---|---|---|
| **Keyword rule (`ERROR` / `FATAL` / fail / exception)** | Rule baseline | Shows what simple rules miss |
| **PCA** | Unsupervised, statistical | Sessions far from the main pattern of normal event counts (squared residual, Xu et al. 2009) |
| **Isolation Forest** | Unsupervised ML | Sessions that are easy to isolate from the rest |
| **Logistic Regression / Decision Tree** | Supervised baseline | Uses labels; shows the upper bound when labels are available |
| **LSTM next-event prediction (DeepLog-style)** | Semi-supervised, deep learning | Trained on normal sequences to predict the next event; a session is anomalous if actual events are repeatedly unlikely |
| **Transformer-based sequence model** | Deep learning | Captures longer-range dependencies between events |

Deep models are included because many failures appear only in the *order* of events, which count vectors ignore. They are compared against simpler methods rather than assumed to be better.

Unsupervised models train on all training sessions **without labels**, as they would in production. Each model's alert threshold is the one that maximises F1 on the validation split. It is then frozen and applied to the test split. PR-AUC is reported alongside because it does not depend on a threshold.

---

## 📏 Evaluation Metrics

| Metric | Why |
|---|---|
| **Precision** | Of the alerts raised, how many were real problems; low precision causes alert fatigue |
| **Recall** | Of the real problems, how many were caught |
| **F1-score** | Balance of precision and recall; the standard metric in log anomaly research |
| **PR-AUC** | Threshold-independent performance under heavy class imbalance |
| **False alarms per day** | Operational view of how noisy the system would be in production |
| **Detection delay** | How soon after an anomaly begins it is flagged (Phase 3, streaming) |
| **Parsing accuracy** | Checks that log parsing is correct, since parsing errors carry through to every model |

Accuracy is not used as a primary metric: with ~97% normal sessions, predicting "normal" for everything already scores ~97%.

---

## 📈 Results

*Chronological 60/20/20 split; metrics on the test split.*

### HDFS

<!-- HDFS_RESULTS -->

### BGL

*To be updated.*

---

## 📁 Project Structure

```
system-log-anomaly-detection/
├── data/                    # not committed
│   ├── raw/                 # Loghub datasets
│   ├── parsed/              # Structured logs and templates
│   └── sequences/           # Grouped sessions / windows
├── artifacts/               # Fitted parser and models (not committed)
├── results/                 # Metric tables per dataset
├── notebooks/
├── src/
│   ├── config.py            # Paths and split fractions
│   ├── data/                # Dataset download
│   ├── parsing/             # Line formats + Drain-based parsing
│   ├── grouping/            # Sessions, time windows, chronological split
│   ├── features/            # Event count vectors
│   ├── models/              # Model definitions and training
│   ├── evaluation/          # Metrics and parsing accuracy
│   ├── streaming/           # Log replay simulator (Phase 3)
│   └── api/                 # FastAPI app (Phase 3)
├── dashboard/               # Streamlit app (Phase 3)
├── tests/
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/coder-zen/system-log-anomaly-detection.git
cd system-log-anomaly-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data (HDFS ≈ 187 MB zipped, BGL ≈ 57 MB)
python -m src.data.download --dataset hdfs

# Parse logs and build sequences
python -m src.parsing.parse_logs --dataset hdfs
python -m src.grouping.build_sequences --dataset hdfs

# Train and evaluate models (writes results/hdfs_classical.md)
python -m src.models.train --dataset hdfs

# Parsing accuracy on Loghub's labelled 2k sample
python -m src.data.download --dataset hdfs --sample
python -m src.evaluation.parsing_accuracy --dataset hdfs

# Tests
pytest
```

For BGL, use `--dataset bgl`. `build_sequences` also accepts `--window-minutes` / `--step-minutes`, and `train` accepts `--purge-overlap` for sliding windows.

---

## 🚀 Deployment (Phase 3–4)

- **Streaming simulation:** a replay script feeds historical logs line by line, as they would arrive from a live system. Lines are parsed online with `drain3`, grouped into sessions or windows, and scored as each group completes.
- **FastAPI:** a `/score` endpoint that accepts a batch of log lines and returns anomaly scores, flagged sessions, and the events that contributed most to each flag.
- **Streamlit dashboard:** an anomaly timeline, the most frequent and newly seen log templates, and a drill-down view showing the raw log lines behind each alert.
- **MLflow:** experiment tracking and model comparison.
- **Docker:** consistent environment for all components.
- **GitHub Actions:** automated tests on every push.

A message queue such as Kafka is **not** used in this version. The replay script is enough to demonstrate the streaming logic, and adding a queue would increase complexity without changing what the project demonstrates.

---

## 🧗 Challenges & Learnings

- **PCA is fragile when anomalies are consistent.** If many anomalous sessions share the same pattern, that pattern carries enough variance to become a principal component, and PCA then treats it as normal. Models trained on unlabeled (contaminated) data must be checked for this.
- **Parsing is the base of the pipeline.** Masking IDs, IPs and numbers before Drain is what makes HDFS templates almost perfectly accurate. BGL's free-text kernel messages are harder.

Still to be investigated:
- Handling log messages that never appeared during training
- Keeping false alarms low enough to be useful

---

## 🔮 Future Work

- Handling **log format changes** after software updates, which is a common cause of model degradation in production
- Root-cause hints: pointing to the most likely component or event behind an anomaly
- Combining logs with system metrics such as CPU, memory, and latency
- Testing on additional Loghub datasets (Thunderbird, OpenStack, Hadoop)
