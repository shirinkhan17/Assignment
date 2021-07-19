import random
import msvcrt

while True:
 print("""Welcome to rock paper scissers game, Instructions are given below:
       if you want to play rock type 'r' or 'rock'
       if you want to play paper type 'p' or 'paper'
       if you want to play scissors type 's' or 'scissors'""")
 user = input("\nEnter a choice: ")
 possible_outcomes = ["rock", "paper", "scissors"]
 computer = random.choice(possible_outcomes)
 user=user.lower()
 if user=="r":
  user="rock"
 elif user=="p":
  user="paper"
 elif user=="s":
  user="scissors"
 print(f"\nYou play {user}, computer play {computer}.\n")
 if user in possible_outcomes:
    if user == computer:
        print(f"Both players play {user}. It's a tie!")
    elif user == "rock":
        if computer == "scissors":
            print("Rock smashes scissors! You win!")
        else:
            print("Paper covers rock! You lose.")
    elif user == "paper":
        if computer == "rock":
            print("Paper covers rock! You win!")
        else:
            print("Scissors cuts paper! You lose.")
    elif user == "scissors":
        if computer == "paper":
            print("Scissors cuts paper! You win!")
        else:
            print("Rock smashes scissors! You lose.")
    
 else:
  print(f"That's not a valid play. You play {user}, check your spelling! User should play rock (r), paper (p) or scissors (s)")
     
 print("\n\nWant to play again? Press y or type any other character: ")
 play_again = msvcrt.getch()
 if play_again != b'y':
   print("You are exiting the game.")
   break 