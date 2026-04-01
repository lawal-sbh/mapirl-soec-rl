class SOECReactor:
    def __init__(self, temperature, pressure, flow_rate):
        self.temperature = temperature
        self.pressure = pressure
        self.flow_rate = flow_rate
    def calculate_efficiency (self):
        return self.flow_rate / (self.temperature * self.pressure)
    def is_safe(self):
        return 800 <= self.temperature <= 1200 and 1 <= self.pressure <= 3
    def __str__(self):
        return f"Reactor: T={self.temperature}K, P={self.pressure}bar, Flow={self.flow_rate} mol/s"
    
reactor1 = SOECReactor(1000, 1.5, 0.10)
reactor2 = SOECReactor(900, 2.0, 0.20)
reactor3 = SOECReactor(1500, 0.5, 0.30)

print(reactor1.is_safe())
print(reactor2.is_safe())
print(reactor3.is_safe())
print(reactor1)
print(reactor2)
print(reactor3)

print(reactor1.calculate_efficiency())
print(reactor2.calculate_efficiency())


print(reactor1.temperature)
print(reactor2.temperature)

