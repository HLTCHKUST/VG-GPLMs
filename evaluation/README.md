# Compute the BLEU,MENTOR and CIDEr scores
+ Install nlg-eval: ```pip install git+https://github.com/Maluuba/nlg-eval.git@master```
+ Setup: ```nlg-eval --setup```
+ Calculate scores: ```nlg-eval --hypothesis=summaries.txt --references=references.txt```

# Compute ROUGE scores
+ Follow git+https://github.com/tagucci/pythonrouge.git to instal ```rouge```
+ Calculate scores: ```python python_rouge.py -c=summaries.txt -r=references.txt```

# Compute Content F1 scores
+ Install meteor: ```git@github.com:cmu-mtlab/meteor.git```
+ Move ```meteor-1.5.jar``` to the  ```meteor``` folder: ```mv meteor-1.5.jar ./meteor/meteor-1.5.jar```
+ ```cd meteor```
+ Use metoer to make alignments: ```java -Xmx2G -jar meteor-1.5.jar summaries.txt references.txt -writeAlignments -f align.out```
+ Calculate scores: ```python calculate_content_f1.py -a=align.out```