# named-entity-recognizer

This is a Named Entity Recognition module to identify Person Name, Location and 
Organization for given input text. The classification model adopts 
the [Stanford Named Entity Recognizer (Stanford NER)](https://nlp.stanford.edu/software/CRF-NER.html).

## Prerequisites

1. Download the Stanford Named Entity Recognizer from 
https://nlp.stanford.edu/software/CRF-NER.html#Download
It is a 151MB zipped file (mainly consisting of classifier data objects). 
If you unpack that file, you should have everything needed for English NER 
(or use as a general CRF). It includes batch files for running under Windows 
or Unix/Linux/MacOSX, a simple GUI, and the ability to run as a server. 
Stanford NER requires Java version 1.8+.

2. Required Python packages: **nltk** (version 3 or above)


## Implementation

For example, perform Named Entity Recognition for the input sentence:
```
Tim went to JP Morgan office in New York.
```

**Import packages and set up**
```
import os
import nltk
from namedentityrecognizer import NER

java_path = ".../jdk1.8.0_171/bin/java.exe" # path to java.exe in machine
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java(java_path)
```
**Performing Named Entity Recognition**
```
input_text = "Tim went to JP Morgan office in New York."

# path to classification model in Stanford NER
cls_model_path = ".../stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz"
# path to .jar file in Stanford NER
st_jar_file_path = ".../stanford-ner/stanford-ner.jar"

classifier = NER(cls_model_path, st_jar_file_path) # initialize an Named Entity Recognizer object
classifier.ner_tag(input_text)
```

**Output**
```
"<Person>Tim</Person> went to <Organization>JP Morgan</Organization> office in <Location>New York</Location>."
```
