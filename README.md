<h1 align="center">CoGenAV: Contrastive-Generative Audio-Visual Representation Learning</h1>


---
#### 🚀 Project Overview  
CoGenAV is a framework for audio-visual representation learning based on **Contrastive-Generative Synchronization**, designed to learn efficient and generalizable audio-visual representations through multimodal alignment of speech, lip movements, and text. The model performs exceptionally well across multiple audio-visual tasks, including:  
- **Audio-Visual Speech Recognition (AVSR)**  
- **Visual Speech Recognition (VSR)**  
- **Audio-Visual Speech Enhancement and Separation (AVSE/AVSS)**  
- **Active Speaker Detection (ASD)**  

---

## 🏗️ Framework

<p align="center">
<img src="https://github.com/user-attachments/assets/fd679c40-41f7-48f8-b4c1-dd30e97f1831" width=100%>
<p>

The left panel depicts the Audio-Visual Feature Representation framework and the Contrastive-Generative Synchronization Training methodology. For generative synchronization, we design a Feature Adaptation Module and employ a [frozen pre-trained ASR model](https://github.com/openai/whisper) as the Speech Recognition (SR) head. The right panel demonstrates the application of CoGenAV to diverse downstream tasks, including Visual Speech Recognition (VSR), Audio-Visual Speech Recognition (AVSR), Audio-Visual Speech Separation (AVSS), Audio-Visual Speech Enhancement (AVSE), and Active Speaker Detection (ASD).

---
#### Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   #Need to ensure that whisper and fairseq is installed
   pip install -U openai-whisper
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install --editable ./

2. Infer and eval:
    - step1. Download the model weights from [modelscope Anonymous CoGenav](https://www.modelscope.cn/models/Anonymou0425/Anonymous_CoGenav/files) to the "weights" folder.
    - step2. Pre-processing LRS2 follow [auto_avsr](https://github.com/mpc001/auto_avsr/tree/main/preparation) and save to data.
    - step3. infer and eval data with infer_vsr_avsr.py  and infer_avse_avss.py
   ```bash
    python infer_vsr_avsr.py --input_type cogenav_av --model_size large  --cogenav_ckpt weights/large_cogenav.pt 
    python infer_avse_avss.py --task_type avse

3. Infer CoGenAV for VSR/AVSR :
   ```python
    import whisper
    from whisper.model import AudioEncoder
    from infer_vsr_avsr import cogenav_forward
    from models.cogenav import CoGenAV
    # Override the Whisper encoder's forward function
    AudioEncoder.forward = cogenav_forward
    # Load CoGenAV model
    cogenav = CoGenAV(cfg_file="config/base.yaml", model_tensor="weights/base_cogenav.pt")
    # Load Whisper model as SR_Head
    SR_Head = whisper.load_model("small", download_root="weights/whisper/")
    SR_Head.encoder.adapter = cogenav.adapter.half()
    # Prepare input using CoGenAV
    input_ids = cogenav(video, audio).permute(0, 2, 1)  # For cogenav_av
    # input_ids = cogenav(video, None).permute(0, 2, 1)  # For cogenav_v
    # input_ids = cogenav(None, audio).permute(0, 2, 1)  # For cogenav_a
    # input_ids = audio  # For whisper_a
    # Decode using Whisper model
    result = whisper.decode(SR_Head, input_ids, options)[0]

4. Infer CoGenAV for AVSS/AVSE :
   ```python
    from models.cogenav import CoGenAV
    from  models.sepformer import build_Sepformer
    # Load CoGenAV model
    cogenav = CoGenAV(cfg_file="config/base.yaml", model_tensor="weights/base_cogenav.pt")
    # Load sepformer model as avss/avse head
    sepformer_head = build_Sepformer().cuda()
    # sep speech with lip feature from mix wav
    lip_feature = cogenav(video, None,use_upsampler=False)
    sep_wav = sepformer_head.forward(audio_mix, lip_feature)
   
## 🎬 Demo
### Demo For AVSR/VSR
<table class="center">
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bold;">
      AVSR/VSR
    </td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/e44e4606-9ef0-4fc7-a1e0-0add000f8e5f" controls preload></video>
      <video src="https://github.com/user-attachments/assets/6c0cfe05-e82e-4b05-bd07-f4e0ebf2375f" controls preload></video>
      <video src="https://github.com/user-attachments/assets/d1190323-dd31-4a74-b2f7-25ce3ec72c35" controls preload></video>
    </td>
  </tr>
</table>

### Demo For AVSS/AVSE

<table style="width:100%; text-align:center;">
  <tr>
    <td colspan="2" style="font-weight: bold; font-size: 1.5em; text-align: center;">
      AVSS(Audio-Visual Speech Separation)
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/13181ace-bb1e-4a6a-97b5-440caa1c93ef" controls preload></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/24a128fb-9686-4c48-955c-8f48c98847a8" controls preload></video>
    </td>
  </tr>
</table>

<table style="width:100%; text-align:center;">
  <tr>
    <td colspan="4" style="font-weight: bold; font-size: 1.5em; text-align: center;">
      AVSE(Audio-Visual Speech Enhancement)
    </td>
  </tr>
  <tr>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/bd7205e8-4eac-4f24-b5a3-251c35b35429" controls preload></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/3101da59-b535-43dc-b58f-8d62625a4b8b" controls preload></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/7f2011bf-ad67-4a67-b7b9-619e3bf04692" controls preload></video>
    </td>
    <td width="25%">
      <video src="https://github.com/user-attachments/assets/e37e19d6-9a63-422b-b200-d827b4e9b317" controls preload></video>
    </td>
  </tr>
</table>

> **Note:** The data in the demo comes from open-source data.
---
## Result
### CoGenAV Base for VSR/AVSR
| Size        | SR Head        | Modalities | VSR  | AVSR@noise | AVSR@clean | AVSR with sft whisper @clean |
|-------------|----------------|------------|------|------------|------------|------------|
|     -        | Whisper medium  | A          | -    | 34.2       | 6.4        | 1.5                          |
| **Base**    | Whisper small   | AV         | 24.8 | 5.2        | 2.5        | -                            |
| **Large**   | Whisper medium  | AV         | **20.4** | **2.6**        | 1.8        | **1.27**             |
> **Note:** VSR/AVSR results on LRS2. The evaluation metric used is WER, and the results are obtained from training conducted solely on the LRS2 dataset.

### CoGenAV Base for AVSS/AVSE
| Task        | SS Head       | Test Dataset    | SI-SNRi | SDRi | PESQ  |
|-------------|----------------|------------------|---------|------|-------|
| **AVSS**    | AV-Sepformer   | mix_2_spk_tt     | 15.7    | 16.0 | 3.23  |
| **AVSE**    | AV-Sepformer   | lrs2_test+noise  | 8.3     | 9.0  | 2.56  |

> **Note:** AVSS/AVSE results on LRS2. These metrics represent the average values for all speakers in each test set, where larger SI-SNRi, SDRi, and PESQ are better.

### CoGenAV Base for ASD
| Task        | SD Head       | Test Dataset    | mAP |
|-------------|----------------|------------------|---------|
| **ASD**    | LRASD   | Talkies     | 96.3    | 
