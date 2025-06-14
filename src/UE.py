import random
# from config import param
from config.param import ParamConfig
# param = ParamConfig({})


class UE:
    def __init__(self, id, x, param= None):
        if param != None:
            self.param = param
        else:   
            self.param = ParamConfig({})

        self.direction = 1 # 1 from right to left, -1 for left to right
        self.x = x
        self.y = random.randint(242, 258)
        self.id = id
        self.nearby_eNBs = []
        self.associated_eNB = None
        self.velocity = random.uniform(self.param.MIN_SPEED, self.param.MAX_SPEED)
    
    def __str__(self):
        return "UE %s located at (%s, %s) with %s m/s" % (self.id, self.x, self.y, self.velocity)

    def get_id(self):
        return self.id
        
    def get_eNBs(self):
        return self.nearby_eNBs

    def get_position(self):
        return (self.x, self.y)
    
    def get_velocity(self):
        return self.velocity
    
    def get_associated_eNB(self):
        return self.associated_eNB

    def set_associated_eNB(self, eNB):
        self.associated_eNB = eNB

    def set_eNBs(self, eNBs):
        self.nearby_eNBs = eNBs

    def set_position(self, position):
        self.x = position[0]
        self.y = position[1]
    
    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_direction(self, direction):
        self.direction = direction
    
    def move(self, ticker):
        self.x += self.direction * self.velocity * ticker.ticker_duration *   random.uniform(0.9, 1.1)  # Randomized movement
        ticker.tick()
