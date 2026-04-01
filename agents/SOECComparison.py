class   SOECComparison:
    def __init__(self, results):
        self.results = results
    def best_algorithm(self):
        if not self.results:
            return None
        best = self.results[0]
        for results in self.results:
            if  results["safety_rate"] > best["safety_rate"]:
                best = results
        return best["algorithm"]
    def summary(self):
        print("-" * 55)
        print(f"{'Algorithm':<12} {'Safety Rate':>12} {'V_cell':>10} {'dT':>10}")
        print("-" * 55)
        for result in self.results:
            print(f"{result['algorithm']:<12} {result['safety_rate']:>11}% {result['v_cell']:>10} {result['dT']:>10}")
        print("-" * 55)

results =[{"algorithm": "SAC", "safety_rate": 100.0, "v_cell": 1.28, "dT": -16.1},
          {"algorithm": "TD3", "safety_rate": 100.0, "v_cell": 1.28, "dT": -15.8},
]
comp = SOECComparison(results)
print("=" * 55)
print("   SAC vs TD3 — SOEC Safety Comparison")
print("=" * 55)
print("Best algorithm:", comp.best_algorithm())
print()
comp.summary()        


       
    

    