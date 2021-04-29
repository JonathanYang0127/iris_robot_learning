fixed_tasks = [
    0,
    90,
    180,
    270,
    45,
    135,
    225,
    315,
    22.5,
    112.5,
    202.5,
    292.5,
    67.5,
    157.5,
    247.5,
    337.5,
]
new_tasks = []
delta = 360 / 32
for task in fixed_tasks:
    new_tasks.append(task + delta)

tasks = fixed_tasks + new_tasks
print('delta', delta)
for t in sorted(tasks):
    print(t)

tasks = [{'goal': t} for t in tasks]
import pickle
pickle.dump(tasks, open(f"ant_32_tasks.pkl", "wb"))
