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
        self.n = 0
    
    def start(self):
        self.time = time.time()
        sys.stdout.write('\r')
        sys.stdout.write(f"{self.title} [%-{self.width}s] %f%% estimated: %fs" % (self.design*self.count, self.percentage, time.time()-self.time))
        sys.stdout.flush()


    def update(self, amount:float): # percentage
        self.percentage = amount

        self.n += 1
        if (self.n == 10):
            sys.stdout.write('\r')
            sys.stdout.write(f"{self.title} [%-{self.width-1}s] %f%% estimated: %fs" % (self.design*self.count, self.percentage, time.time()-self.time))
            sys.stdout.flush()
            self.n = 0
        if (self.percentage - self.updated >= self.diff):
            self.count += 1
            self.updated = self.count*self.diff

            if (self.count == self.width):
                self.end()

    def end(self):
        sys.stdout.write("\n")
