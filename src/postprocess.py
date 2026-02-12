import numpy as np


def decode_notes(frame_probs, onset_probs, frame_thresh=0.5, onset_thresh=0.5, 
                 min_duration_frames=5, fps=31.25):
    notes = []
    num_frames, num_pitches = frame_probs.shape
    
    for pitch in range(num_pitches):
        frames = frame_probs[:, pitch] > frame_thresh
        onsets = onset_probs[:, pitch] > onset_thresh
        
        # get onset pos
        onset_frames = np.where(onsets)[0]
        
        for onset_frame in onset_frames:
            if not frames[onset_frame]:
                continue # onset must also have frame activation
            
            # offset from where activation ends
            offset_frame = onset_frame + 1
            while offset_frame < num_frames and frames[offset_frame]:
                offset_frame += 1
            
            # applying minimum duration to notes
            if offset_frame - onset_frame < min_duration_frames:
                offset_frame = onset_frame + min_duration_frames
            
            # converting into sec
            onset_time = onset_frame / fps
            offset_time = offset_frame / fps
            
            # converting from pith index into midi
            midi_pitch = pitch + 21
            
            notes.append((midi_pitch, onset_time, offset_time))
    
    return notes



