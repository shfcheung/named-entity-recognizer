# -*- coding: utf-8 -*-
"""
@author: Sam Cheung

This is a Named Entity Recognition module to identify Person Name, Location and 
Organization for given input text, which the entity classification model adopts 
the Stanford Named Entity Recognizer (Stanford NER).
Reference: https://nlp.stanford.edu/software/CRF-NER.html

Example input: Tim went to JP Morgan office in New York.
Output: '<Person>Tim</Person> went to <Organization>JP Morgan</Organization> 
        office in <Location>New York</Location>.'

==============================================================================
Prerequisites

1. Download the Stanford Named Entity Recognizer from 
https://nlp.stanford.edu/software/CRF-NER.html#Download

It is a 151MB zipped file (mainly consisting of classifier data objects). 
If you unpack that file, you should have everything needed for English NER 
(or use as a general CRF). It includes batch files for running under Windows 
or Unix/Linux/MacOSX, a simple GUI, and the ability to run as a server. 
Stanford NER requires Java v1.8+.

2. Required Python packages: nltk (version 3 or above)

3. Before running the module, run the following codes in Python: 
import os
import nltk
java_path = ".../jdk1.8.0_171/bin/java.exe" # path to java.exe in machine
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java(java_path)
==============================================================================

"""

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


class NER(object):
    
    """
    A class for named-entity recognizer
    INPUTS:
        cls_model_path: the path to the pre-trained model for NER tagging, it
                        is in the downloaded Stanford Named Entity Recognizer
        st_jar_file_path: the path to stanford tagger jar file, it
                          is in the downloaded Stanford Named Entity Recognizer
    """
    
    def __init__(self, cls_model_path, st_jar_file_path):
        self.st = StanfordNERTagger(cls_model_path, 
                                    st_jar_file_path,
                                    encoding='utf-8')
    
    def ner_tag(self, input_text):
        """
        A function to identify Person Name, Location and Organization given input
        text.
        
        INPUTS: 
            input_text: A string object
        RETURNS: 
            ner_tagged_text = string object with entity tags inserted
                       
        """
        
        tokenized_text = word_tokenize(input_text) # tokenize text
        # apply Stanford NER to tag the tokenized text, the output  
        # "classified_text" is a list of (token, entity) tuples 
        # e.g. [('Jim', 'PERSON'),('bought', 'O'),('300', 'O'),
        #      ('shares', 'O'),('of', 'O'),('Acme', 'ORGANIZATION'),
        #      ('Corp.', 'ORGANIZATION'),('in', 'O'),('2006', 'O'),('.', 'O')]
        classified_tokens = self.st.tag(tokenized_text)
        # transform the list of (token, entity) tuples to a string object
        ner_tagged_text = self.__format_tagged_tokens(classified_tokens)
        
        return ner_tagged_text
    
    def __format_tagged_tokens(self, tuple_list):
        
        """
        A function to transform the outputs (in the form of a list of tuples) 
        from Stanford NER to string object. 
        
        INPUTS:
            tuple_list: a list of (token, entity) tuples
        RETURNS:
            output_text: string object            
        """
        
        # unzip the tuple_list to token_tuple and ent_tuple respectively
        (token_tuple, ent_tuple) = list(zip(*tuple_list))
        
        # a dictionary that maps index numbers (idx) to tokens, adjacent tokens
        # with the same entity tag are sharing the same index number
        idx_to_token_map = {}  
        # a dictionary that maps index numbers (idx) to entities(ent), adjacent 
        # entity tags (which are the same) are sharing the same index number        
        idx_to_ent_map = {} 
        output_token_list = [] # an empty list to keep the formatted tokens
        
        i = 0
        # add first token to idx_to_token_map with index number 0
        idx_to_token_map[0] = token_tuple[i] 
        # add first entity tag to idx_to_ent_map with index number 0
        idx_to_ent_map[0] = ent_tuple[i].capitalize() 
        
        
        # if the entity tag of the current and previous tokens are the same,
        # concatenate the current token to previous token (separated by a space 
        # character) in 'idx_to_token_map', otherwise (1) increase the index 
        # number in idx_to_token_map by 1 and attach the token to the updated 
        # index number; and (2) increase the index nymber in idx_to_ent_map by
        # 1 and attach the entity tag to the updated index number
        for j in range(1, len(token_tuple)):
            if ent_tuple[j] == ent_tuple[j-1]:
                idx_to_token_map[i] = idx_to_token_map[i] + ' ' + token_tuple[j]
            else:
                i += 1
                idx_to_token_map[i] = token_tuple[j]
                idx_to_ent_map[i] = ent_tuple[j].capitalize()
        
        # formats the tokens belonging to 'Person', 'Organization'
        # or 'Location' entity in the form of '<EntityTag>tokens</EntityTag>'
        for k in range(len(idx_to_token_map)):
            if idx_to_ent_map[k] in ['Person', 'Organization', 'Location']:
                concated_token = '<' + idx_to_ent_map[k] + '>' + \
                idx_to_token_map[k] + '</' + idx_to_ent_map[k] + '>'
                output_token_list.append(concated_token)
            else:
                output_token_list.append(idx_to_token_map[k])
        
        # join the formatted tokens by spcae character
        output_text = ' '.join(output_token_list) 
            
        return output_text
