# Minimal BeamSense workflow template

This file gives a command template for one complete hardware-loop run.  
All placeholders must be replaced before use.

## 0. Required placeholders

```text
<REPO_ROOT>
<PROJECT_ROOT>
<MATLAB_EXE>
<UBUNTU_IP>
<WIN10_IP>
<WIN11_IP>
<RATE_A>
<RATE_B>
<DURATION_SEC>
<SESSION_NAME>
<LAYOUT_NAME>
<PREHEAT_SEC>
<CAPTURE_SEC>
<ORIENTATION_NOTE>
<SESSION_NOTES>
```

Typical formal-session timing:

```text
empty       0-30 s
one_person  40-70 s
two_person  80-110 s
```

For this type of run, use a capture duration long enough to cover the full schedule plus buffer, for example `<CAPTURE_SEC>=120`.

## 1. Start iperf3 servers on Ubuntu

Run these on the Ubuntu receiver/sniffer machine, preferably in two separate terminals:

```bash
iperf3 -s -B <UBUNTU_IP> -p 5201
iperf3 -s -B <UBUNTU_IP> -p 5202
```

## 2. Start iperf3 clients on Win10 and Win11

Run this on the Win10 endpoint:

```powershell
iperf3.exe -c <UBUNTU_IP> -B <WIN10_IP> -u -b <RATE_A> -t <DURATION_SEC> -i 1 -p 5201
```

Run this on the Win11 endpoint:

```powershell
iperf3.exe -c <UBUNTU_IP> -B <WIN11_IP> -u -b <RATE_B> -t <DURATION_SEC> -i 1 -p 5202
```

## 3. Start Ubuntu capture / QC / push

Run this on the Ubuntu sniffer after the foreground traffic is already running:

```bash
cd <REPO_ROOT>/ubuntu

./run_capture_qc_push_2.sh \
  <SESSION_NAME> \
  <LAYOUT_NAME> \
  "iperf A=<RATE_A> B=<RATE_B>" \
  "rtax52_a,rtax52_b,rtax52_c" \
  <PREHEAT_SEC> \
  <CAPTURE_SEC> \
  "<ORIENTATION_NOTE>" \
  "<SESSION_NOTES>"
```

Start the physical action schedule when the script prints:

```text
[INFO] CAPTURE START NOW
```

## 4. Prepare the session on Windows

After the Ubuntu script pushes the session to the Windows inbox, run the preparation script from the repository root:

```powershell
cd <REPO_ROOT>

.\prepare_session_v2.ps1
```

If your local version of the script exposes parameters, pass your session name and project paths explicitly. The sanitised scripts use placeholders such as `<PROJECT_ROOT>` that must be replaced.

## 5. Run the Windows/MATLAB processing pipeline

```powershell
cd <REPO_ROOT>

.\run_session_pipeline_v2.ps1
```

This step calls MATLAB scripts for BFA extraction and batch generation. Replace `<MATLAB_EXE>` and `<PROJECT_ROOT>` in the script before running.

## 6. Train or evaluate the Keras model

Training scripts are in:

```text
python/states_e12p_bmain_keras/
```

Typical files:

```text
dataGenerator_states.py
train_states_baseline.py
eval_states_model.py
```

Use `train_states_baseline.py` for training and `eval_states_model.py` for held-out evaluation.

## 7. Run blind inference and plot the timeline

Inference scripts are in:

```text
python/inference/
```

Typical sequence:

```powershell
python python/inference/create_infer_segments.py --session-name <SESSION_NAME> --segments-root <SEGMENTS_ROOT> --duration-sec <DURATION_SEC>

python python/inference/infer_keras_session.py `
  --infer-csv <INFER_CSV> `
  --model-path <MODEL_PATH> `
  --label-map-json <LABEL_MAP_JSON> `
  --data-generator-dir python/states_e12p_bmain_keras `
  --out-dir <INFERENCE_OUT_DIR> `
  --session-name <SESSION_NAME> `
  --vote-sec 5 `
  --window-interval 0.1

python python/inference/plot_inference_timeline.py `
  --sample-csv <INFERENCE_OUT_DIR>/sample_predictions.csv `
  --interval-csv <INFERENCE_OUT_DIR>/interval_predictions_5s.csv `
  --out <INFERENCE_OUT_DIR>/prediction_timeline_5s.png
```

## 8. Safety reminder

Before committing any new configuration or result file, check that it does not contain:

```text
real MAC addresses
private IP addresses
usernames
passwords
SMB credentials
raw capture files
trained model files
large generated datasets
```
