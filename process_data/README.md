# Data Processing Database

## Overview
This codebase contains scripts and utilities for converting between various musical data formats, including ABC notation, MusicXML, MIDI, and MTF (MIDI Text Format). Additionally, it includes a script for summarizing music metadata, which is represented in JSON format containing textual information, using the OpenAI GPT-4 API. The GPT-4 model processes this metadata to generate concise summaries in multiple languages to boost multilingual MIR. These tools are designed to facilitate the transformation and manipulation of musical files, as well as to provide concise multilingual summaries of music metadata for use with CLaMP 2.


## About ABC notation
### Standard ABC Notation  
ABC notation (sheet music), a text-based sheet music representation like stave notation, is theory-oriented and ideal for presenting complex musical concepts to musicians for study and analysis. Standard ABC notation encodes each voice separately, which often results in corresponding bars being spaced far apart. This separation makes it difficult for models to accurately understand the interactions between voices in sheet music that are meant to align musically.

Example Standard ABC notation representation:  
```
%%score { 1 | 2 }
L:1/8
Q:1/4=120
M:3/4
K:G
V:1 treble nm="Piano" snm="Pno."
V:2 bass
V:1
!mf!"^Allegro" d2 (GA Bc | d2) .G2 .G2 |]
V:2
[G,B,D]4 A,2 | B,6 |]
```

### Interleaved ABC Notation  
In contrast, interleaved ABC notation effectively aligns multi-track music by integrating multiple voices of the same bar into a single line, ensuring that all parts remain synchronized. This format combines voices in-line and tags each bar with its corresponding voice (e.g., `[V:1]` for treble and `[V:2]` for bass). By directly aligning related bars, interleaved ABC notation enhances the model’s understanding of how different voices interact within the same bar.

Below is the same data optimized with M3 encoding, where each bar or header corresponds to a patch:  
```
%%score { 1 | 2 }
L:1/8
Q:1/4=120
M:3/4
K:G
V:1 treble nm="Piano" snm="Pno."
V:2 bass
[V:1]!mf!"^Allegro" d2 (GA Bc|[V:2][G,B,D]4 A,2|
[V:1]d2) .G2 .G2|][V:2]B,6|]
```

## About MTF
### Raw MIDI Messages  
MIDI (performance data) precisely encodes performance information related to timing and dynamics, thus suitable for music production and live performance. Raw MIDI messages contain essential musical instructions and metadata, extracted directly from a MIDI file. These include events like note on/off, tempo changes, key signatures, and control changes, which define how the music is performed. The [mido library](https://mido.readthedocs.io/) allows for reading these messages in their native format, as seen below. Each message can include multiple parameters, making the output comprehensive but sometimes redundant.  

```
MetaMessage ('time_signature', numerator=3, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)
MetaMessage('key_signature', key='G', time=0)
MetaMessage('set_tempo', tempo=500000, time=0)
control_change channel=0 control=121 value=0 time=0
program_change channel=0 program=0 time=0
control_change channel=0 control=7 value=100 time=0
control_change channel=0 control=10 value=64 time=0
control_change channel=0 control=91 value=0 time=0
control_change channel=0 control=93 value=0 time=0
MetaMessage('midi_port', port=0, time=0)
note_on channel=0 note=74 velocity=80 time=0
MetaMessage('key_signature', key='G', time=0)
MetaMessage('midi_port', port=0, time=0)
note_on channel=0 note=55 velocity=80 time=0
note_on channel=0 note=59 velocity=80 time=0
note_on channel=0 note=62 velocity=80 time=0
note_on channel=0 note=74 velocity=0 time=455
note_on channel=0 note=67 velocity=80 time=25
note_on channel=0 note=67 velocity=0 time=239
note_on channel=0 note=69 velocity=80 time=1
note_on channel=0 note=55 velocity=0 time=191
note_on channel=0 note=59 velocity=0 time=0
note_on channel=0 note=62 velocity=0 time=0
note_on channel=0 note=69 velocity=0 time=48
note_on channel=0 note=71 velocity=80 time=1
note_on channel=0 note=57 velocity=80 time=0
note_on channel=0 note=71 velocity=0 time=239
note_on channel=0 note=72 velocity=80 time=1
note_on channel=0 note=57 velocity=0 time=215
note_on channel=0 note=72 velocity=0 time=24
note_on channel=0 note=74 velocity=80 time=1
note_on channel=0 note=59 velocity=80 time=0
note_on channel=0 note=74 velocity=0 time=455
note_on channel=0 note=67 velocity=80 time=25
note_on channel=0 note=67 velocity=0 time=239
note_on channel=0 note=67 velocity=80 time=241
note_on channel=0 note=67 velocity=0 time=239
note_on channel=0 note=59 velocity=0 time=168
MetaMessage('end_of_track', time=1)
```
### MIDI Text Format (MTF)  
The MIDI Text Format (MTF) provides a structured, textual representation of MIDI data that preserves all original information without loss. Each MIDI message is accurately represented, allowing full reconstruction, ensuring no musical nuances are overlooked during conversion.  

To generate MTF, the mido library reads raw MIDI messages from MIDI files. The output retains all essential information but can be lengthy and redundant. To simplify the representation, parameter values are read in a fixed order and separated by spaces. For example, the raw time signature message, which contains several parameters—numerator, denominator, clocks per click, notated 32nd notes per beat, and time—is represented in MTF as:  

```
time_signature 3 4 24 8 0
```

Other messages, such as control changes and note events, follow a similar compact format while preserving all relevant musical details. This structured simplification improves computational performance and maintains precise control over musical elements, including timing and dynamics.  

Example MTF representation:  
```
ticks_per_beat 480
time_signature 3 4 24 8 0
key_signature G 0
set_tempo 500000 0
control_change 0 0 121 0
program_change 0 0 0
control_change 0 0 7 100
control_change 0 0 10 64
control_change 0 0 91 0
control_change 0 0 93 0
midi_port 0 0
note_on 0 0 74 80
key_signature G 0
midi_port 0 0
note_on 0 0 55 80
note_on 0 0 59 80
note_on 0 0 62 80
note_on 455 0 74 0
note_on 25 0 67 80
note_on 239 0 67 0
note_on 1 0 69 80
note_on 191 0 55 0
note_on 0 0 59 0
note_on 0 0 62 0
note_on 48 0 69 0
note_on 1 0 71 80
note_on 0 0 57 80
note_on 239 0 71 0
note_on 1 0 72 80
note_on 215 0 57 0
note_on 24 0 72 0
note_on 1 0 74 80
note_on 0 0 59 80
note_on 455 0 74 0
note_on 25 0 67 80
note_on 239 0 67 0
note_on 241 0 67 80
note_on 239 0 67 0
note_on 168 0 59 0
end_of_track 1
```
For simplicity, `ticks_per_beat`, though originally an attribute of MIDI objects in mido, is included as the first message at the beginning of the MTF representation.

### M3-Encoded MTF  
When processed using M3 encoding, consecutive messages of the same type that fit within a 64-character limit (the patch size of M3) are combined into a single line. Only the first message in each group specifies the type, with subsequent messages listing only the parameter values separated by tabs. This further simplifies the representation and improves processing efficiency.  

Below is the same data optimized with M3 encoding, where each line corresponds to a patch:  
```
ticks_per_beat 480
time_signature 3 4 24 8 0
key_signature G 0
set_tempo 500000 0
control_change 0 0 121 0
program_change 0 0 0
control_change 0 0 7 100\t0 0 10 64\t0 0 91 0\t0 0 93 0
midi_port 0 0
note_on 0 0 74 80
key_signature G 0
midi_port 0 0
note_on 0 0 55 80\t0 0 59 80\t0 0 62 80\t455 0 74 0\t25 0 67 80
note_on 239 0 67 0\t1 0 69 80\t191 0 55 0\t0 0 59 0\t0 0 62 0
note_on 48 0 69 0\t1 0 71 80\t0 0 57 80\t239 0 71 0\t1 0 72 80
note_on 215 0 57 0\t24 0 72 0\t1 0 74 80\t0 0 59 80\t455 0 74 0
note_on 25 0 67 80\t239 0 67 0\t0 67 80\t239 0 67 0\t168 0 59 0
end_of_track 1
```

By reducing redundancy, M3 encoding ensures improved computational performance while maintaining precise timing and musical control, making it an ideal choice for efficient MIDI processing.

## Repository Structure
The `process_data/` folder includes the following scripts and utility files:

### 1. **Conversion Scripts**

#### `batch_abc2xml.py`
- **Purpose**: Converts ABC notation files into MusicXML format.
- **Input**: Directory of interleaved ABC files (modify the `input_dir` variable in the code).
- **Output**: MusicXML files saved in a newly created `_xml` directory.
- **Logging**: Errors are logged to `logs/abc2xml_error_log.txt`.

#### `batch_xml2abc.py`
- **Purpose**: Converts MusicXML files into standard ABC notation format.
- **Input**: Directory of MusicXML files (e.g., `.xml`, `.mxl`, `.musicxml`) (modify the `input_dir` variable in the code).
- **Output**: Standard ABC files saved in a newly created `_abc` directory.
- **Logging**: Errors are logged to `logs/xml2abc_error_log.txt`.

#### `batch_interleaved_abc.py`
- **Purpose**: Processes standard ABC notation files into interleaved ABC notation.
- **Input**: Directory of ABC files (modify the `input_dir` variable in the code).
- **Output**: Interleaved ABC files saved in a newly created `_interleaved` directory.
- **Logging**: Any processing errors are printed to the console.

#### `batch_midi2mtf.py`
- **Purpose**: Converts MIDI files into MIDI Text Format (MTF).
- **Input**: Directory of MIDI files (e.g., `.mid`, `.midi`) (modify the `input_dir` variable in the code).
- **Output**: MTF files saved in a newly created `_mtf` directory.
- **Logging**: Errors are logged to `logs/midi2mtf_error_log.txt`.
- **Note**: The script includes an `m3_compatible` variable, which is set to `True` by default. When `True`, the conversion omits messages whose parameters are strings or lists to eliminate potential natural language information. This ensures that the converted MTF files align with the data format used for training the M3 and CLaMP 2 pretrained weights.

#### `batch_mtf2midi.py`
- **Purpose**: Converts MTF files into MIDI format.
- **Input**: Directory of MTF files (modify the `input_dir` variable in the code).
- **Output**: MIDI files saved in a newly created `_midi` directory.
- **Logging**: Errors are logged to `logs/mtf2midi_error_log.txt`.

### 2. **Summarization Script**

#### `gpt4_summarize.py`
- **Purpose**: Utilizes the OpenAI GPT-4 API to generate concise summaries of music metadata in multiple languages. The script filters out any entries that lack sufficient musical information to ensure meaningful summaries are produced.
- **Input**: Directory of JSON files containing music metadata (modify the `input_dir` variable in the code). For any missing metadata fields, the corresponding keys can be set to `None`. Each JSON file corresponds to a single musical composition and can be linked to both ABC notation and MTF formats. Here’s an example of the required metadata format:

  ```json
  {
    "title": "Hard Times Come Again No More",
    "composer": "Stephen Foster",
    "genres": ["Children's Music", "Folk"],
    "description": "\"Hard Times Come Again No More\" (sometimes referred to as \"Hard Times\") is an American parlor song written by Stephen Foster, reflecting themes of sorrow and hope.",
    "lyrics": "Let us pause in life's pleasures and count its many tears,\nWhile we all sup sorrow with the poor;\nThere's a song that will linger forever in our ears;\nOh! Hard times come again no more.\n\nChorus:\n'Tis the song, the sigh of the weary,\nHard Times, hard times, come again no more.\nMany days you have lingered around my cabin door;\nOh! Hard times come again no more.\n\nWhile we seek mirth and beauty and music light and gay,\nThere are frail forms fainting at the door;\nThough their voices are silent, their pleading looks will say\nOh! Hard times come again no more.\nChorus\n\nThere's a pale weeping maiden who toils her life away,\nWith a worn heart whose better days are o'er:\nThough her voice would be merry, 'tis sighing all the day,\nOh! Hard times come again no more.\nChorus\n\n'Tis a sigh that is wafted across the troubled wave,\n'Tis a wail that is heard upon the shore\n'Tis a dirge that is murmured around the lowly grave\nOh! Hard times come again no more.\nChorus",
    "tags": ["folk", "traditional", "bluegrass", "nostalgic", "heartfelt", "acoustic", "melancholic", "storytelling", "American roots", "resilience"],
    "ensembles": ["Folk Ensemble"],
    "instruments": ["Vocal", "Violin", "Tin whistle", "Guitar", "Banjo", "Tambourine"],
    "filepaths": [
      "abc/American_Music/Folk_Traditions/19th_Century/Stephen_Foster/Hard_Times_Come_Again_No_More.abc",
      "mtf/American_Music/Folk_Traditions/19th_Century/Stephen_Foster/Hard_Times_Come_Again_No_More.mtf"
    ]
  }
  ```

- **Output**: JSON files containing structured summaries in both English and a randomly selected non-English language, chosen from a selection of 100 different non-English languages (in this case, Simplified Chinese). Here’s an example of the expected output format:

```json
{
  "title": "Hard Times Come Again No More",
  "composer": "Stephen Foster",
  "genres": ["Children's Music", "Folk"],
  "description": "\"Hard Times Come Again No More\" (sometimes referred to as \"Hard Times\") is an American parlor song written by Stephen Foster, reflecting themes of sorrow and hope.",
  "lyrics": "Let us pause in life's pleasures and count its many tears,\nWhile we all sup sorrow with the poor;\nThere's a song that will linger forever in our ears;\nOh! Hard times come again no more.\n\nChorus:\n'Tis the song, the sigh of the weary,\nHard Times, hard times, come again no more.\nMany days you have lingered around my cabin door;\nOh! Hard times come again no more.\n\nWhile we seek mirth and beauty and music light and gay,\nThere are frail forms fainting at the door;\nThough their voices are silent, their pleading looks will say\nOh! Hard times come again no more.\nChorus\n\nThere's a pale weeping maiden who toils her life away,\nWith a worn heart whose better days are o'er:\nThough her voice would be merry, 'tis sighing all the day,\nOh! Hard times come again no more.\nChorus\n\n'Tis a sigh that is wafted across the troubled wave,\n'Tis a wail that is heard upon the shore\n'Tis a dirge that is murmured around the lowly grave\nOh! Hard times come again no more.\nChorus",
  "tags": ["folk", "traditional", "bluegrass", "nostalgic", "heartfelt", "acoustic", "melancholic", "storytelling", "American roots", "resilience"],
  "ensembles": ["Folk Ensemble"],
  "instruments": ["Vocal", "Violin", "Tin whistle", "Guitar", "Banjo", "Tambourine"],
  "summary_en": "\"Hard Times Come Again No More,\" composed by Stephen Foster, is a poignant American parlor song that explores themes of sorrow and hope. The lyrics reflect on the contrast between life's pleasures and its hardships, inviting listeners to acknowledge both joy and suffering. With a heartfelt chorus that repeats the line \"Hard times come again no more,\" the song resonates with nostalgia and resilience. It is often performed by folk ensembles and features a variety of instruments, including vocals, violin, guitar, and banjo, encapsulating the spirit of American roots music.",
  "summary_nen": {
    "language": "Chinese (Simplified)",
    "summary": "《艰难时光再无来临》是斯蒂芬·福斯特创作的一首感人至深的美国小歌厅歌曲，探讨了悲伤与希望的主题。歌词展现了生活的乐趣与艰辛之间的对比，邀请听众去感受快乐与痛苦的交织。歌曲中那句反复吟唱的“艰难时光再无来临”深情地表达了怀旧与坚韧。它常常由民谣乐队演奏，伴随着人声、小提琴、吉他和班卓琴等多种乐器，生动地展现了美国根源音乐的独特魅力。"
  },
  "filepaths": [
    "abc/American_Music/Folk_Traditions/19th_Century/Stephen_Foster/Hard_Times_Come_Again_No_More.abc",
    "mtf/American_Music/Folk_Traditions/19th_Century/Stephen_Foster/Hard_Times_Come_Again_No_More.mtf"
  ]
}
```

- **Logging**: Errors are logged to `logs/gpt4_summarize_error_log.txt`.

### 3. **Utilities**
- **`utils/`**: Contains utility files required for the conversion processes.

## Usage
To use the scripts, modify the `input_dir` variable in each script to point to the directory containing your input files. Then run the script from the command line. Below are example commands for each script:

### Example Commands
```bash
# Modify the input_dir variable in the script before running
python batch_abc2xml.py
python batch_xml2abc.py
python batch_interleaved_abc.py
python batch_midi2mtf.py
python batch_mtf2midi.py
python gpt4_summarize.py
```

### Execution Order
To achieve specific conversions, follow the order below:

1. **To obtain interleaved ABC notation**:
   - First, run `batch_xml2abc.py` to convert MusicXML files to ABC notation.
   - Then, run `batch_interleaved_abc.py` to process the ABC files into interleaved ABC notation.

2. **To obtain MTF**:
   - Run `batch_midi2mtf.py` to convert MIDI files into MTF.

3. **To convert interleaved ABC back to XML**:
   - Run `batch_xml2abc.py` on the interleaved ABC files to convert them back to MusicXML format.

4. **To convert MTF back to MIDI**:
   - Run `batch_mtf2midi.py` to convert MTF files back to MIDI format.

5. **To summarize music metadata**:
   - Run `gpt4_summarize.py` to generate summaries for the music metadata files in JSON format. This assumes you have a directory of JSON files that includes a `filepaths` key, which connects to the corresponding interleaved ABC and MTF files.

### Parameters
To run the scripts, you need to configure the following parameters:

- **`input_dir`**: This variable should be set to the directory containing the input files to be processed (such as ABC, MusicXML, MIDI, MTF, or JSON files), which is shared across all scripts.

In addition to **`input_dir`**, the following parameters are specific to certain scripts:

- **`m3_compatible`** (specific to `batch_midi2mtf.py`):
  - Default is `True`, which omits messages with parameters that are strings or lists to avoid including potential natural language information.
  - Setting this to `False` retains all MIDI messages, which is crucial for those planning to retrain models on custom datasets or needing precise MIDI reproduction.

For **`gpt4_summarize.py`**, you also need to configure these parameters:

1. **`base_url`**: The base URL for the OpenAI API, used to initialize the client.  
2. **`api_key`**: Your API key for authenticating requests, required for client initialization.  
3. **`model`**: The GPT-4 model to use, specified when generating summaries.

  **Important**: When `m3_compatible` is set to `True`, the conversion back from MTF to MIDI using `batch_mtf2midi.py` may produce MIDI files that do not exactly match the original MIDI files. This discrepancy is unexpected; however, retraining both M3 and CLaMP 2 to address this issue would require approximately 6000 hours of H800 GPU hours. Considering that M3 and CLaMP 2 have already achieved state-of-the-art results on MIDI tasks, we have opted not to retrain. Therefore, if consistency with original MIDI files is critical for your application, it is advisable to set `m3_compatible` to `False`.
