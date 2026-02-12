import numpy as np
import pretty_midi

MIN_MIDI_PITCH = 21 # A0
MAX_MIDI_PITCH = 108 # C8
NUM_MIDI_PITCHES = 88 # n keys

class MIDILabels:
    def __init__(self, cfg):
        self.sample_rate = cfg.sample_rate
        self.hop_length = cfg.hop_length
        self.fps = cfg.sample_rate / cfg.hop_length

    def __call__(self, midi_path, num_frames):
        midi = pretty_midi.PrettyMIDI(midi_path)

        frame_labels = np.zeros((num_frames, NUM_MIDI_PITCHES), dtype=np.float32)
        onset_labels = np.zeros((num_frames, NUM_MIDI_PITCHES), dtype=np.float32)

        for instrument in midi.instruments:
            if instrument.is_drum: continue # shoulnt be in maestro, more if i want to extend later

            for note in instrument.notes:
                if note.pitch < MIN_MIDI_PITCH or note.pitch > MAX_MIDI_PITCH:
                    continue

                pitch_i = note.pitch - MIN_MIDI_PITCH
                start_frame = int(note.start * self.fps) # TODO maybe round instead?
                end_frame = int(note.end * self.fps)

                # clamping if outside valid range
                start_frame = max(0, start_frame)
                end_frame = min(num_frames, end_frame)

                frame_labels[start_frame:end_frame, pitch_i] = 1.0

                if start_frame < num_frames: # should not be outside spectrogram
                    onset_labels[start_frame, pitch_i] = 1.0

        return frame_labels, onset_labels


