import math
import copy
import time
import sys
import random
import copy
#Smallest supersequence found so far
best_string = None
#Length of smallest supersequence
best_string_length = math.inf
starting_time = None
ticks = 0
ticks_max = 1000
#Adds substring to end of string
def add_substring(string, substring):
    if substring in string:
        return string
    length = len(substring)
    while(True):
        if string[-length:] == substring[:length]:
            string += substring[length:]
            break
        length -= 1
    return string

def permutation_string(permutation, substrings):
    string = ''
    for number in permutation:
        string = add_substring(string, substrings[number])
    return string

def swap_permutation(permutation, swap_number):
    new_permutation = copy.deepcopy(permutation)
    new_permutation[swap_number], new_permutation[swap_number + 1] = new_permutation[swap_number + 1], new_permutation[swap_number]
    return new_permutation

#DFS search for smallest supersequence. Runs for a max of 10 seconds
def complete_DFS(string, substrings, search_time=10):
    global best_string
    global best_string_length
    global ticks
    ticks += 1
    if ticks == ticks_max:
        current_time = time.time()
        if current_time - starting_time > search_time:
            output_best()
        ticks = 0
    #Calculate if string generated is the smallest found
    if not substrings:
        string_length = len(string)
        if string_length < best_string_length:
            best_string = string
            best_string_length = string_length
    substring_tries = [(substring, add_substring(string, substring)) for substring in substrings]
    #Sort in order of length 
    substring_tries = sorted(substring_tries, key=lambda x: len(x[1]))
    #string is too long. At sometime we will have to add this substring!
    if not substring_tries or len(substring_tries[-1][1]) >= best_string_length:
        return
    #DFS search
    while substring_tries:
        substring = substring_tries[0]
        substring_tries.remove(substring)
        substrings.remove(substring[0])
        complete_DFS(substring[1], substrings, search_time)
        substrings.append(substring[0])
def local_search(substrings, iterations=1000, search_time=10, print_string=True):
    global best_string
    global best_string_length
    global ticks
    num_substrings = len(substrings)
    current_permutation = range(num_substrings)
    while(True):
        current_permutation = random.sample(current_permutation, num_substrings)
        for _ in range(iterations):
            ticks += 1
            if ticks == ticks_max:
                current_time = time.time()
                if current_time - starting_time > search_time:
                    if print_string == True:
                        output_best()
                    ticks = 0
                    return
                ticks = 0
            random_choice = random.randrange(2)
            #Random move
            if random_choice == 0:
                swap_number = random.randrange(0, num_substrings - 1)
                current_permutation = swap_permutation(current_permutation, swap_number)
            else:
                permutation_swaps = [None for _ in range(num_substrings - 1)]
                new_strings = [None for _ in range(num_substrings - 1)]
                for num in range(num_substrings - 1):
                    permutation_swaps[num] = swap_permutation(current_permutation, num)
                    new_strings[num] = permutation_string(current_permutation, substrings)
                permutation_lengths = [len(swap) for swap in new_strings]
                min_index = permutation_lengths.index(min(permutation_lengths))
                current_permutation = swap_permutation(current_permutation, min_index)
            new_string = permutation_string(current_permutation, substrings)
            if len(new_string) < best_string_length:
                best_string = new_string
                best_string_length = len(new_string)



#Outputs best string found so far and exits program
def output_best():
    print("Best sequence found: ", best_string)
    print("Sequence Length: ", best_string_length)
    sys.exit()

#Read input and run DFS on subsequences
def main():
    global starting_time
    filename = sys.argv[1]
    with open(filename) as f:
        _ = int(f.readline())
        _ = int(f.readline())
        subsequences = []
        sequence = None
        while(sequence != ''):
            sequence = f.readline()
            if sequence != '':
                #Drop newline
                subsequences.append(sequence[:-1])

    starting_time = time.time()
    local_search(subsequences, search_time=5, print_string=False)
    starting_time = time.time()
    complete_DFS('', subsequences, search_time=5)
    output_best()      
if __name__ == '__main__':
    main()