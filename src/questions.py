# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:13:49 2022

@author: class

part of the JuPy Project
2020

CC BY-NC-ND 4.0

Original Version of the create_multipleChoice_widget() seems to be from: 
    
 https://github.com/dingandrew/Zumi-Python-Lessons/blob/master/Quiz_Generator.py   
"""



import numpy as np
import ipywidgets as widgets
import random as rnd
from IPython.display import display
from IPython.display import clear_output
from IPython.display import YouTubeVideo
from IPython.display import Markdown
from IPython.display import Code


def create_multipleChoice_widget_new_feedback(descriptionText, descriptionCode, options, correct_answer, feedback_dic, last=0):
    rnd.shuffle(options)
    
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False
    )
    
    description_out = widgets.Output()
    with description_out:
        if len(descriptionText)>0:
                m=Markdown(descriptionText)
                display(m)
        if len(descriptionCode)>0:
                #c=Code(descriptionCode,language='python')
                c=Markdown("```python\n"+descriptionCode+"\n```")
                display(c)
        
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.value)
        answerText=options[a]
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Richtig." + '\x1b[0m' +"\n" #green color
        elif answerText in feedback_dic.keys():
            s = '\x1b[5;30;41m Falsch. '+feedback_dic[answerText] + '\x1b[0m' +"\n" #red color  
        else:
            s = '\x1b[5;30;41m' + "Falsch. " + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="überprüfen")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])


