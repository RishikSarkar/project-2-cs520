# Define all the bots here

class Bot1:
    def __init__(self, position):
        self.name = "Bot1"
        self.position = position
        self.crew_found = 0
    
    def move(self):
        return