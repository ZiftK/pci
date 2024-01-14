




def ahorcado():

    up = "-------------\n" 
    line =  "|\n"*6
    ls = "|" + " "*11 + "|"

    str = [
        "-------------\n"+"|\n"*6,
        "-------------\n"+"|" + " "*11 + "|\n" +"|\n"*5,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" +"|\n"*4,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*11 + "|\n"+"|\n"*3,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*11 + "|/\n"+"|\n"*3,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*10 + "\\|/\n"+"|\n"*3,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*10 + "\\|/\n"+ "|" + " "*11 + "|\n"+"|\n"*2,
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*10 + "\\|/\n"+ "|" + " "*11 + "|\n"+ "|" + " "*10 + "/\n" + "|\n",
        "-------------\n"+"|" + " "*11 + "|\n" + "|" + " "*11 +"O\n" + "|" +" "*10 + "\\|/\n"+ "|" + " "*11 + "|\n"+ "|" + " "*10 + "/ \\\n" + "|\n",
        ]
    
    shower = str[0]
    error_count = 0
    max_error = 8

    print("\n\n")
    word = input("Ingresa una palabra> ")

    print("\n\n")

    size = " _ "*len(word)

    while error_count < max_error:
        
        print("")
        print(shower)
        print(size)

        letter = input("Ingresa una letra> ")

        #* La letra no está
        if not (letter.lower() in word.lower()):
            print("\nERROR!!!!")

            error_count += 1

            shower = str[error_count]

        #* La letra si está
        else:
            
            for char in size:
                if char == letter:
                    
                    pass
                pass
            pass
        print("\n\n")

    print(shower)
    print("PERDEDOR :C")

if __name__ == "__main__":
    
    ahorcado()
    pass