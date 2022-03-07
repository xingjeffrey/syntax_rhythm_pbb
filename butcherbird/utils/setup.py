from pathlib2 import Path
import pathlib2
from butcherbird.utils.paths import ensure_dir, DATA_DIR

import pandas as pd
import json
from tqdm.autonotebook import tqdm

from praatio import tgio

import librosa
import wave
from scipy.io.wavfile import read


def get_file_id (wav_path): 
    '''
    Return unique butcherbird recording ID
    '''
    return wav_path.__str__().split('/')[-1].rstrip('.wav').rstrip('.WAV')

def get_seg (wav_path, tg_loc):
    '''
    description:
    Find segmentation textgrid file paths of a given wav
    
    parameters:
    wav_path: target wav file
    
    '''
    ## get recording id
    wav_id = get_file_id(wav_path).split('.')[0]
    
    ## use glob to find all paths of textgrids
    return list(tg_loc.glob(wav_id + "*"))

def get_bird_nm (tg_path):
    '''
    description:
    Extract name of bird in annotation file
    
    parameters:
    tg_path: path of specified textgrid
    
    '''
    return get_file_id(tg_path).split('_')[-1].split('.')[0]

def get_samplerate(wav_path):
    '''
    description:
    find sample rate of a given wav
    
    parameters:
    wav_path = target wave file location
    
    '''
    print(wav_path)
    
    return read(wav_path)[0]

##Praat Related

def tg_to_phrase_note (tg_path_or_obj):
    '''
    description:
    Extract textgrid information into a list format
    
    parameters:
    tg_path_or_obj: path of specified textgrid, or opened textgrid
    
    '''
    ## skeleton output array
    tier_output = []
    
    ## if is path, open a textgrid obj, if is already obj do nothing
    if isinstance(tg_path_or_obj, pathlib2.PosixPath):
        tg_trgt = tgio.openTextgrid(tg_path_or_obj)
    else:
        tg_trgt = tg_path_or_obj
        
    ## for every tier
    for tier in tg_trgt.tierNameList:
        ## extract information from that tier and then add to output skeleton
        curr_tier = tg_trgt.tierDict[tier].entryList
        tier_output = tier_output + [curr_tier]
        
    return tier_output

## write function to transform textgrids into dataframes
def song_to_df (wav_path, tg_loc):
    '''
    description:
    Parse annotated butcherbird wavs into dataframes, preserving phrase/note structure from textgrids
    
    parameters:
    wav_path = target recording of this dataframe (Path)
    
    '''
    ## helpful prints
    print(wav_path)
    
    ## make empty container, where each note is an unique entry
    song_df = pd.DataFrame(
        columns=[
            'wav_nm',
            'tg_nm',
            'bird_nm',
            'phrase_nb', 
            'phrase_strt', 
            'phrase_end',
            'phrase_len',
            'note_cnt',
            'note_nb',
            'note_strt',
            'note_end',
            'note_len',
        ]
    )
    
    ## immutable constant for wav name
    wav_nm = get_file_id(wav_path)
    
    ##SANITY
    ##print(wav_nm)
    
    ## load in all textgrid data associated with target wav
    tg_trgt_loc = get_seg(wav_path, tg_loc)
    tg_trgt = [[tg_to_phrase_note(tg_loc), tg_loc] for tg_loc in tg_trgt_loc]
    '''
    tg_trgt contains all textgrids associated with a wav:
        first level (tg_bird): different textgrids/birds
        second level: phrase/note data (container), textgrid path
    tg_trgt = [
                [
                    [
                        [[[phrase start],[phrase end],[p]], ......],
                        [[[note start],[note end],[n]], ......]
                    ], 
                    textgrid path
                ],
                ......
              ]
    
    '''
    
    ## for every bird/textgrid associated with a wav file
    for tg_bird in tqdm(tg_trgt):
        ## load in proper textgrid object
        tg_container = tgio.openTextgrid(tg_bird[1])
        
        ## load in textgrid/bird name
        tg_nm = get_file_id(tg_bird[1])
        bird_nm = get_bird_nm(tg_bird[1])
        
        ## load in phrases and notes
        phrase_container = tg_bird[0][0]
        note_container = tg_bird[0][1]

        ## start filling info

        ## counters
        phrase_nb = -1

        ## grab all the phrases in sample tg
        for phrase in tqdm(phrase_container):

            ## at current phrase, mark phrase number
            phrase_nb = phrase_nb + 1
            
            ## get current phrase info
            curr_phrase = phrase
            phrase_strt = curr_phrase[0]
            phrase_end = curr_phrase[1]
            phrase_len = phrase_end - phrase_strt
            
            ##SANITY
            ##print(phrase_strt)
            
            ## crop notes that belong to this phrase
            curr_phrase_crop = tg_container.crop(phrase_strt, phrase_end, mode = 'strict', rebaseToZero = False)
            note_crop = tg_to_phrase_note(curr_phrase_crop)[1]
            
            ## add note count of each phrase
            note_cnt = len(note_crop)
            
            ##SANITY
            ##print(phrase_nb)
            
            ## counters
            note_nb = -1
            
            ## for every note in this phrase, create an entry into the DataFrame
            for curr_note in note_crop:
                
                ## at current note, mark note number
                note_nb = note_nb + 1
                
                ## get current note info
                note_strt = curr_note[0]
                note_end = curr_note[1]
                note_len = note_end - note_strt
                
                ##SANITY
                ##print(note_nb)
                
                ## With all the information collected, append into the DataFrame as an observation
                song_df.loc[len(song_df)] = [
                    wav_nm,
                    tg_nm,
                    bird_nm,
                    phrase_nb, 
                    phrase_strt, 
                    phrase_end,
                    phrase_len,
                    note_cnt,
                    note_nb,
                    note_strt,
                    note_end,
                    note_len,
                ]
    
    ## once everything has been filled out, index by wav_nm, return the DataFrame
    return song_df.set_index('wav_nm')

def wav_to_json (df_cohort, DATASET_ID, DT_ID, wav_path, tg_loc):
    '''
    description:
    Parse wav into a json files using textgrid information in df_cohort0
    
    parameters:
    wav_path = target recording of this json (Path)
    
    '''
    
    ## make json dictionary
    json_dict = {}
    
    ## add species
    json_dict["species"] = "Cracticus nigrogularis"
    json_dict["common_name"] = "Pied butcherbird"
    
    ## add wav info
    json_dict["wav_loc"] = wav_path.__str__()
    json_dict["wav_nm"] = get_file_id(wav_path)
    json_dict["lengths_s"] = librosa.get_duration(filename = wav_path)
    json_dict["samplerate_hz"] = get_samplerate(wav_path).__int__()
    
    ## grab appropriate sections of df_cohort0_sum
    df_filtered = df_cohort.filter(like = get_file_id(wav_path), axis = 0)
    
    ## associated birds and textgrids 
    tg_asso = get_seg(wav_path, tg_loc)
    
    ## individuals container
    json_dict["indvs"] = {}
    
    ## iterate through every individual textgrid present for this recording
    for tg in tg_asso:
        
        ## get specified bird and textgrid attributes
        tg_loc = tg
        tg_nm = get_file_id(tg_loc)
        bird_nm = get_bird_nm(tg_nm)
        json_dict["indvs"][bird_nm] = {"tg_loc":tg_loc.__str__(),
                                       "tg_nm":tg_nm, 
                                       "units":{
                                           "phrases": {
                                               "nb": df_filtered['phrase_nb'].values.astype("int").tolist(),
                                               "strt": df_filtered['phrase_strt'].values.astype("float64").tolist(),
                                               "end": df_filtered['phrase_end'].values.astype("float64").tolist(),
                                               "len": df_filtered['phrase_len'].values.astype("float64").tolist(),
                                           },
                                           "notes": {
                                               "cnt": df_filtered['note_cnt'].values.astype("int").tolist(),
                                               "nb": df_filtered['note_nb'].values.astype("int").tolist(),
                                               "strt": df_filtered['note_strt'].values.astype("float64").tolist(),
                                               "end": df_filtered['note_end'].values.astype("float64").tolist(),
                                               "len": df_filtered['note_len'].values.astype("float64").tolist()
                                           }
                                       }
                                      }
                                       
    '''
    ##SANITY DRAIN
    print(type(json_dict["wav_loc"]))
    print(type(json_dict["wav_nm"]))
    print(type(json_dict["lengths_s"]))
    print(type(json_dict["samplerate_hz"]))
    print(type(json_dict["indvs"][bird_nm]["tg_loc"]))
    print(type(json_dict["indvs"][bird_nm]["tg_nm"]))
    print(type(json_dict["indvs"][bird_nm]["notes"]["phrase_nb"][0]))
    print(type(json_dict["indvs"][bird_nm]["notes"]["phrase_strt"][0]))
    
    '''
    
    ## when everything is collected, dump dict into json
    json_txt = json.dumps(json_dict, indent = 2)
    
    wav_stem = get_file_id(wav_path)
    json_out = (
        DATA_DIR / "interim" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )
    
    ## save json
    ## remember to ensure safe operations later!!
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(Path(json_out),'w'))
    
    return json_txt