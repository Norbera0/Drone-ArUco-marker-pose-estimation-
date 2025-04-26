import keyboard
import time

class KeyboardHandler:
    def __init__(self, speed=100):
        self.speed = speed
        self.key_states = {
            "UP": False,
            "DOWN": False,
            "LEFT": False,
            "RIGHT": False,
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False
        }
        
    def init(self):
        """Initialize keyboard hooks"""
        for key in self.key_states.keys():
            keyboard.on_press_key(key, lambda e, k=key: self._on_key_press(k))
            keyboard.on_release_key(key, lambda e, k=key: self._on_key_release(k))
    
    def _on_key_press(self, key):
        """Handle key press events"""
        if key in self.key_states:
            self.key_states[key] = True
    
    def _on_key_release(self, key):
        """Handle key release events"""
        if key in self.key_states:
            self.key_states[key] = False
    
    def get_control_inputs(self):
        """Get current control inputs based on key states"""
        lr, fb, ud, yv = 0, 0, 0, 0
        
        if self.key_states["UP"]: ud = self.speed
        elif self.key_states["DOWN"]: ud = -self.speed
        if self.key_states["w"]: fb = self.speed
        elif self.key_states["s"]: fb = -self.speed
        if self.key_states["a"]: lr = -self.speed
        elif self.key_states["d"]: lr = self.speed
        if self.key_states["LEFT"]: yv = -self.speed
        elif self.key_states["RIGHT"]: yv = self.speed
        
        return [lr, fb, ud, yv]
    
    def should_land(self):
        """Check if landing command is active"""
        return self.key_states["q"]
    
    def should_takeoff(self):
        """Check if takeoff command is active"""
        return self.key_states["e"] 