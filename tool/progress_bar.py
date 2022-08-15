import time, sys

box = "█"

class bar:
    def __init__(self, width=40, design="-", title=""):
        self.width = width
        self.design = design
        self.percentage = 0
        self.updated = 0
        self.diff = 100/width
        self.count = 0
        self.title = title
    
    def start(self):
        sys.stdout.write(f"{self.title} [%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.width+1)) # return to start of line, after '['
        self.time = time.time()


    def update(self, amount:float): # percentage
        self.percentage = amount

        if (self.percentage - self.updated >= self.diff):
            sys.stdout.write(self.design)
            sys.stdout.flush()
            self.count += 1
            self.updated = self.count*self.diff

            if (self.count == self.width):
                self.end()

    def end(self):
        sys.stdout.write(f"] estimated: {round(time.time()-self.time, 2)}\n") # this ends the progress bar"
