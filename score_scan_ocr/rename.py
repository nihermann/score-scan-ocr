import os
from utx import osx

if __name__ == '__main__':
    for path, dirs, files in os.walk("splits"):
        for file in files:
            if file.endswith(".pdf"):
                if False:
                    file_path = os.path.join(path, file)
                    # print(file_path)
                    part = file.split(" - ")[-1]
                    os.rename(file_path, os.path.join(path, "El Gigante de Hierro - Ferran - " + part))
                else:
                    base = "Danse Satanique - Kosmicki - "
                    instr = file.split("Danse Satanique ")[-1]
                    for i, char in enumerate(instr):
                        try:
                            if str(char).isnumeric() or (instr[i-1].islower() and char.isupper()):
                                base += " "
                        except Exception as e:
                            print(e)
                        base += char
                    file_path = os.path.join(path, file)
                    os.rename(file_path, os.path.join(path, base))
