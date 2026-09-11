# DeepCTC: End-to-End Automatic Speech Recognition

DeepCTC is a robust, end-to-end Automatic Speech Recognition (ASR) system built with TensorFlow. It is designed to swiftly and accurately transcribe raw speech audio into text. The architecture is heavily inspired by DeepSpeech2, utilizing a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) trained with Connectionist Temporal Classification (CTC) loss.

## Features
- **End-to-End Transcription:** Directly maps audio sequences to text sequences.
- **Audio Preprocessing:** Transforms raw WAV audio into spectrograms via Short-Time Fourier Transform (STFT), enabling robust frequency-domain feature learning.
- **DeepSpeech2 Inspired Architecture:**
  - **Conv2D Layers:** Extracts spatial acoustic features from the spectrograms.
  - **Bi-Directional GRUs:** Models the temporal context and sequence dependencies of the speech.
  - **Dense Output Layers:** Predicts the character probabilities at each time step.
- **CTC Loss:** Uses Connectionist Temporal Classification loss to align the unsegmented audio sequences with the text transcriptions.

## Dataset
The model is trained on the **[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)**, a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

## Performance
- **Transcription Accuracy:** 91.6%
- **Final Validation Loss:** 12.7

## How to Run

### Option 1: Google Colab (Recommended)
Because training an ASR model requires significant compute power and the notebook is already configured for Colab environments, this is the easiest way to run the project.

1. Open `Automatic_Speech_Recognition.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type** and set the Hardware accelerator to **GPU**.
3. Go to **Runtime > Run all** (or execute cells sequentially with `Shift + Enter`). The notebook will automatically download the 2.6GB LJ Speech dataset and begin preprocessing and training.

### Option 2: Local Execution
To run this project locally, a dedicated NVIDIA GPU with CUDA configured for TensorFlow is highly recommended.

1. Clone or download the repository to your local machine.
2. Install the necessary dependencies:
   ```bash
   pip install jupyter tensorflow numpy matplotlib
   ```
3. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
4. Open `Automatic_Speech_Recognition.ipynb`.
5. **Note:** You may need to manually adjust the data download and extraction paths in the first few cells (e.g., removing Colab-specific paths like `/content/dataset/`) to match your local Windows or Linux directory structure.
6. Run the cells sequentially.
