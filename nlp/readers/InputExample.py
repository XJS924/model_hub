from typing import Union, List

class InputExample:
    
    def __init__(self, guid: str= '', texts :List[str]= None, label: Union[int, float]= 0):

        self.guid= guid
        self.texts = texts
        self.label = label 
    
    def __str__(self):
        return f"<InputExample> label: {self.label}, texts: {self.texts}"