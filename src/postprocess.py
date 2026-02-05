import numpy as np
import pretty_midi


def decode_notes(frame_probs, onset_probs, frame_thresh=0.5, onset_thresh=0.5, 
                 min_duration_frames=5, fps=31.25):
    notes = []
    num_frames, num_pitches = frame_probs.shape
    
    for pitch in range(num_pitches):
        frames = frame_probs[:, pitch] > frame_thresh
        onsets = onset_probs[:, pitch] > onset_thresh
        
        # Find onset positions
        onset_frames = np.where(onsets)[0]
        
        for onset_frame in onset_frames:
            if not frames[onset_frame]:
                continue  # onset must also have frame activation
            
            # Find offset: where frame activation ends
            offset_frame = onset_frame + 1
            while offset_frame < num_frames and frames[offset_frame]:
                offset_frame += 1
            
            # Apply minimum duration
            if offset_frame - onset_frame < min_duration_frames:
                offset_frame = onset_frame + min_duration_frames
            
            # Convert to time (seconds)
            onset_time = onset_frame / fps
            offset_time = offset_frame / fps
            
            # Convert pitch index to MIDI (piano starts at A0 = 21)
            midi_pitch = pitch + 21
            
            notes.append((midi_pitch, onset_time, offset_time))
    
    return notes



def notes_to_midi(notes, output_path):
    
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    for pitch, onset, offset in notes:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=onset,
            end=offset
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(output_path)