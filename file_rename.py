
import os 

def main(): 
    i = 1
    path = 'C:\\Users\\yourpath\\masks'
    os.chdir(path)
    for filename in os.listdir(path): 
        dst = filename[:-5] + ".tiff"
        src = filename
        os.rename(src, dst) 
        i += 1

if __name__ == '__main__': 
    main() 
