from tabulate import tabulate
import os

import solve_cv_stupid
import solve_valid
import solve_cv_valid

def solve_cv_12(image_path):
    return solve_cv_valid.solve(image_path, 12)

def solve_cv_15(image_path):
    return solve_cv_valid.solve(image_path, 15)

table = [["method name"]]
table2 = [["method name"]]
methods = [
    # (solve_cv_stupid.solve, "cv"), 
    # (solve_valid.solve, "valid"), 
    (solve_cv_12, "cv_valid_12"),
    # (solve_cv_15, "cv_valid_15")
]

for filename in os.listdir("data"):
    if os.path.isdir(os.path.join("data", filename)):
        table[0].append(filename)
        table2[0].append(filename)

for unpack in methods:
        table.append([unpack[1]])
        table2.append([unpack[1]])
        method = unpack[0]

        for data in table[0][1:]:
            total = len(os.listdir(os.path.join("data", data)))
            correct = 0
            correct_v = 0
            done = 0

            for filename in os.listdir(os.path.join("data", data)):
                done += 1
                print(f"{unpack[1]}: {data}: {done}/{total}")

                ans_right = [i.split(",") for i in filename[:-4].split(';')]
                ans_guessed = method(os.path.join("data", data, filename))

                if len(ans_right) != len(ans_guessed):
                    continue

                guessed = 0
                for i in ans_right:
                    for j in ans_guessed:
                        if abs(int(i[0]) - j[0]) + abs(int(i[1]) - j[1]) <= 15:
                            guessed += 1
                            correct_v += 1
                            break

                if guessed == len(ans_right):
                    correct += 1

            table[-1].append(correct/total*100)
            table2[-1].append(correct_v/total)

print(tabulate(table, headers="firstrow", tablefmt="grid"))
print()
print(tabulate(table2, headers="firstrow", tablefmt="grid"))