from re import findall
import matplotlib.pyplot as plt


def plot_res_feas_study(file_path,loads,check):
    """
    Will plot the results of the feasibility study. 
    """
    res_string=[]
    lines_counter = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if check == True:
            for line in lines:
                lines_counter+=1
                if lines_counter%2==0:
                    res_string.append(line)
        else:
            for line in lines:
                res_string.append(line)

    
    baseline = []
    concept = []
    for result in res_string:
        numbers = findall(r'\d+\.\d+', result)
        numbers = [float(num) for num in numbers]
        baseline.append(numbers[1])
        concept.append(numbers[0])
            
    plt.xlabel("Load [N]")
    plt.ylabel("Mass [kg]")
    plt.grid()
    
    plt.plot(loads,baseline, '--', label="Oversized beam")
    plt.plot(loads,concept,label="Optimized truss")
    
    plt.legend(loc="upper left")
    
    plt.show()
    

def write_to_text_file(filename, content):
  """
  Writes content to a text file.

  Args:
      filename: The path to the text file.
      content: The string content to be written to the file. End the line with
      \n.
  """

  try:
    with open(filename, "a") as file:
      file.write(content)
      print(f"Data appended to file '{filename}'.")
  except Exception as e:
    print(f"An error occurred while writing to the file: {e}")