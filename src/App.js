import React, { useEffect } from "react";
import "./App.css";

function App() {
  const token = `
  import nltk
word_data = "It originated from the idea that there are readers who prefer learning new skills from the comforts of their drawing rooms"
nltk_tokens = nltk.word_tokenize(word_data)
print (nltk_tokens)

sentence_data = "The First sentence is about Python. The Second: about Django. You can learn Python,Django and Data Ananlysis here."
nltk_tokens = nltk.sent_tokenize(sentence_data)
print (nltk_tokens)
  `;
  const water = `
  class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def Min(a, b):
    if (a < b):
        return a
    return b


def allPossibilities(initialState, maxJugX, maxJugY):
    # Make Memoization Via For Loop
    memoization = [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]
    tree = GeneralTree()

    def helperFunction(state):
        if memoization[state.x][state.y] == 1:
            return
        memoization[state.x][state.y] = 1
        tree.insert(State(state.x, state.y))
        # Fill Left
        helperFunction(State(maxJugX, state.y))
        # Fill Right
        helperFunction(State(state.x, maxJugY))
        # Empty Left
        helperFunction(State(0, state.y))
        # Empty Right
        helperFunction(State(state.x, 0))
        # Transfer From X To Y
        pourXY = min(state.x, maxJugY-state.y)
        helperFunction(State(state.x-pourXY, state.y+pourXY))
        # Transfer From Y To X
        pourYX = min(maxJugX-state.x, state.y)
        helperFunction(State(state.x+pourYX, state.y-pourYX))

    helperFunction(initialState)
    return tree


class Node:
    def __init__(self, state):
        self.state = state
        self.left = None
        self.right = None


class GeneralTree:
    def __init__(self):
        self.head = None

    def insert(self, state):
        newNode = Node(state)
        if (self.head == None):
            self.head = newNode
            return
        queue = []
        queue.append(self.head)
        while len(queue):
            curr = queue.pop(0)
            if (curr.left == None):
                curr.left = newNode
                return
            else:
                queue.append(curr.left)
            if (curr.right == None):
                curr.right = newNode
                return
            else:
                queue.append(curr.right)

    def findPossibility(self, state,maxJugX,maxJugY):
        queue = []
        visited = []
        finalState = None
        queue.append(self.head)
        while len(queue):
            curr = queue.pop(0)
            finalState = curr.state
            visited.append(f"({curr.state.x},{curr.state.y})")
            if (curr.state.x == state.x and curr.state.y == state.y):
                return visited

            if (curr.left != None):
                queue.append(curr.left)
            if (curr.right != None):
                queue.append(curr.right)
            if (curr.state.x == 0 and curr.state.y == 0)or (curr.state.x == maxJugX and curr.state.y ==0) or (curr.state.x == 0 and curr.state.y==maxJugY):
                visited = []
                visited.append(f"({curr.state.x},{curr.state.y})")

        if finalState.x == state.x and finalState.y == state.y:
            return visited
        else:
            return "No Solution"


tree = allPossibilities(State(0, 0), 4, 3)
print(tree.findPossibility(State(2, 3),4,3))


  `;
  const dfs = `
  class Graph:
  def __init__(self):
      self.adjList = {}
  
  def insert(self,fromNode,toNode):
      if fromNode not in self.adjList:
          self.adjList[fromNode] = []
      self.adjList[fromNode].append(toNode)
  
  def dfs(self,node,visited=None):
      if visited == None:
          visited = set()
      print(node)
      visited.add(node)
      for neighbor in self.adjList.get(node,[]):
          if neighbor not in visited:
              self.dfs(neighbor,visited)

g = Graph()
g.insert(1, 2)
g.insert(1, 3)
g.insert(2, 4)
g.insert(2, 5)
g.insert(3, 6)
g.dfs(1)

  `;
  const kanpsack = `
  from itertools import permutations
def generate_permutations(input_array):
    all_permutations = list(permutations(input_array))
    return all_permutations
    

class Item:
    def __init__(self,weight,value):
        self.weight = weight
        self.value = value


def bruteForceKnapSack(items,maxBagSize):
    combinations = []
    for x in range(0,len(items)):
        combinations.append(x)
    allCombinations = generate_permutations(combinations)
    bestSolutionItems = [] 
    bestSolutionValue = -1
    for comb in allCombinations:
        x = 0
        tempBagWeight = 0
        tempBagValue = 0
        tempItems = []
        while tempBagWeight <= maxBagSize:
            if items[comb[x]].weight + tempBagWeight <= maxBagSize:
                tempBagWeight += items[comb[x]].weight
                tempBagValue += items[comb[x]].value
                tempItems.append(x)
            x+=1
            if(x == len(combinations)):
                break
        if tempBagValue > bestSolutionValue:
            bestSolutionItems = tempItems
            bestSolutionValue = tempBagValue
    print(bestSolutionItems)
    print(bestSolutionValue)


#best solution 1 & 2 & 4
Items = [Item(2,12),Item(1,10),Item(3,20),Item(2,15)]
maxBagSize = 5

bruteForceKnapSack(Items,maxBagSize)

  `;
  const xor = `
  def xor_table():
    for a in [0, 1]:
        for b in [0, 1]:
            print(f"{a} XOR {b} = {a ^ b}")

xor_table()

  `;
  const graphColoring = `
nodes = 4

#no. of colors
m = 3

x = [0,0,0,0]

#k is current node
def graphColor(k):
    for c in range(1,m+1):
        if (isSafe(k,c)):
            x[k] = c
            if k+1 < nodes:
                graphColor(k+1)

def isSafe(k , c):
    for i in range(nodes):
        if G[k][i] == 1 and c == x[i]:
            return False
    return True

G = [[1,1,0,1],
     [1,1,1,1],
     [0,1,1,1],
     [1,1,1,1]]
        
graphColor(0)
print(x)`;

  const aStart = `
  import networkx as nx
  import matplotlib.pyplot as plt
  
  def find_shortest_path(grid_data, start, end):
      rows, cols = len(grid_data), len(grid_data[0])
      G = nx.grid_2d_graph(rows, cols)
      for r in range(rows):
          for c in range(cols):
              if grid_data[r][c] == 1:
                  G.remove_node((r, c))
  
      try:
          path = nx.shortest_path(G, source=start, target=end)
          return path
      except nx.NetworkXNoPath:
          return None
  
  grid_data = [
      [0, 0, 0, 1, 0],
      [0, 1, 0, 1, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0]
  ]
  
  start_position = (0, 0)
  end_position = (4, 4)
  
  path = find_shortest_path(grid_data, start_position, end_position)
  
  if path:
      print("Shortest Path:", path)
  else:
      print("No path found.")
  
`;

  const twoPlayer = `
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)


def check_win(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False


def is_full(board):
    for row in board:
        if " " in row:
            return False
    return True


def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    player_turn = "X"

    while True:
        print_board(board)
        while True:
            try:
                row = int(
                    input(f"Enter row (1, 2, or 3) for {player_turn}: ")) - 1
                col = int(
                    input(f"Enter column (1, 2, or 3) for {player_turn}: ")) - 1
                if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == " ":
                    break
                else:
                    print("Invalid input. Try again.")
            except ValueError:
                print("Invalid input. Enter numbers.")

        board[row][col] = player_turn

        if check_win(board, player_turn):
            print_board(board)
            print(f"{player_turn} wins!")
            break
        elif is_full(board):
            print_board(board)
            print("It's a draw!")
            break

        player_turn = "X" if player_turn == "O" else "O"


play_game()

`;

  const singlePLayer = `
import random

def print_board(board):
    [print(" | ".join(row), "\n" + "-" * 5) for row in board]

def is_winner(board, player):
    return any(all(cell == player for cell in row) for row in board) or \
           any(all(board[row][col] == player for row in range(3)) for col in range(3)) or \
           all(board[i][i] == player for i in range(3)) or \
           all(board[i][2 - i] == player for i in range(3))

def is_full(board):
    return all(cell != " " for row in board for cell in row)

def get_random_move(board):
    return random.choice([(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]) if any(" " in row for row in board) else None

def main():
    board, player_turn = [[" "]*3 for _ in range(3)], True
    while True:
        print_board(board)

        if player_turn:
            row, col = map(int, input("Enter row and column (comma-separated): ").split(','))
            if board[row][col] == " ": board[row][col], player_turn = "X", False
            else: print("Cell already occupied. Try again.")
        else:
            print("AI is making a random move...")
            if (move := get_random_move(board)): board[move[0]][move[1]], player_turn = "O", True

        if is_winner(board, "X"): print_board(board), print("Congratulations! You win!"); break
        elif is_winner(board, "O"): print_board(board), print("AI wins! Better luck next time."); break
        elif is_full(board): print_board(board), print("It's a tie!"); break

if __name__ == "__main__":
    main()

`;
  const matLab = `image = imread("Tufaha.jpg");

    %Basic Use Case Question
    %Make Image negative and show all 8 planes
    neg = negativeImage(image);
    bitPlanes(neg); 
    
    
    %Make Image Negative
    function img = negativeImage(image)
        L = 2 ^ 8;
        img = (L - 1) - image;
    end
    
    %Get All 8 Planes Of Image DOES NOT RETURN
    function bitPlanes(image)
        for plane = 1:8
        B = bitget(image, plane);
        subplot(2, 4, plane);
        imshow(255 * B);
        title(['Bit plane ' num2str(plane)]);
        end
    end
    
    %Get Graysacle Version Of Image
    function img = grayScale(image)
        img = rgb2gray(image);
    end
    
    %Remove salt noise
    function img = removeSaltNoise(image)
        filterSize = [5, 5];  
        grayImage = grayScale(image);
        img = medfilt2(grayImage, filterSize);
    end
    
    %Remove pepper noise
    function img = removePepperNoise(image)
        filterSize = [5, 5];  
        grayImage = grayScale(image);
        img = medfilt2(grayImage, filterSize);
    end
    
    %Gls with background
    function img = glsWithBackground(image)
        T1 = 100;
        T2 = 200;
        grayImage = grayScale(image);
        inside_mask = (grayImage >= T1) & (grayImage <= T2);
        output_with_background = grayImage; 
        output_with_background(inside_mask) = 255;
        img = output_with_background;
    end
    
    %Gls without background
    function img = glsWithoutBackground(image)
        T1 = 100;
        T2 = 200;
        grayImage = grayScale(image);
        outside_mask = (grayImage < T1) | (grayImage > T2);
        output_without_background = zeros(size(grayImage), 'uint8');
        output_without_background(outside_mask) = 255;
        img = output_without_background;
    end
    
    %Histogram Equalization DOES NOT RETURN
    function histogramEqualization(image)
        subplot(2,2,1);
        imshow(image)
        subplot(2,2,2);
        imhist(image,64);
        afterImg = histeq(image);
        subplot(2,2,3);
        imshow(afterImg)
        subplot(2,2,4);
        imhist(afterImg,64);
    end
    
    %Erosion
    function img = erosion(image)
        se = strel('disk', 5);
        img = imerode(image, se);
    end
    
    %Open
    function img = open(image)
        se = strel('disk', 5);
        img = imopen(image, se);
    end
    
    %Close
    function img = close(image)
        se = strel('disk', 5);
        img = imclose(image, se);
    end
    
    
    %Dialate
    function img = dialate(image)
        se = strel('disk', 5);
        img = imdilate(image, se);
    end
    
    %Power Log
    function img = powerLog(image)
        input_image = im2double(image);
        gamma = 5
        img = input_image.^gamma;
    end
    
    %Edge Detection
    function img = edgeDetection(image)
        newImage = grayScale(image);
        img = edge(newImage,'Prewitt');
        %img = edge(newImage,'Canny');
    end
    
    %High Pass Butterworth Filter DOES NOT RETURN
    function highPassButterWorth(image)
        grayImage = rgb2gray(image);
        fftImage = fft2(double(grayImage));
        [M, N] = size(fftImage);
        u = 0:(M-1);
        v = 0:(N-1);
        u(u > M/2) = u(u > M/2) - M;
        v(v > N/2) = v(v > N/2) - N;
        [V, U] = meshgrid(v, u);
        D0 = 30; 
        n = 2;   
        H_gaussian_highpass = 1 - exp(-(U.^2 + V.^2) / (2 * D0^2));
        H_butterworth_highpass = 1 ./ (1 + (D0 ./ sqrt(U.^2 + V.^2)).^(2 * n));
        filteredImage_gaussian_highpass = real(ifft2(fftImage .* H_gaussian_highpass));
        filteredImage_butterworth_highpass = real(ifft2(fftImage .* H_butterworth_highpass));
        subplot(2, 2, 1);
        imshow(grayImage);
        title('Original Image');
        subplot(2, 2, 2);
        imshow(filteredImage_gaussian_highpass, []);
        title('Gaussian High Pass Filtered Image');
        subplot(2, 2, 3);
        imshow(filteredImage_butterworth_highpass, []);
        title('Butterworth High Pass Filtered Image');
    end
    
    %Low Pass Filter DOES NOT RETURN
    function lowPassFilter(image)
    grayImage = rgb2gray(image);
    fftImage = fft2(double(grayImage));
    [M, N] = size(fftImage);
    u = 0:(M-1);
    v = 0:(N-1);
    u(u > M/2) = u(u > M/2) - M;
    v(v > N/2) = v(v > N/2) - N;
    [V, U] = meshgrid(v, u);
    D0 = 30; 
    n = 2;   
    H_ideal = double(sqrt(U.^2 + V.^2) <= D0);
    H_gaussian = exp(-(U.^2 + V.^2) / (2 * D0^2));
    H_butterworth = 1 ./ (1 + (sqrt(U.^2 + V.^2) / D0).^(2 * n));
    filteredImage_ideal = real(ifft2(fftImage .* H_ideal));
    filteredImage_gaussian = real(ifft2(fftImage .* H_gaussian));
    filteredImage_butterworth = real(ifft2(fftImage .* H_butterworth));
    subplot(2, 2, 1);
    imshow(grayImage);
    title('Original Image');
    subplot(2, 2, 2);
    imshow(filteredImage_ideal, []);
    title('Ideal Low Pass Filtered Image');
    subplot(2, 2, 3);
    imshow(filteredImage_gaussian, []);
    title('Gaussian Low Pass Filtered Image');
    subplot(2, 2, 4);
    imshow(filteredImage_butterworth, []);
    title('Butterworth Low Pass Filtered Image');
    end
    `;

  //   useEffect(() => {
  //     copyCodeToClipboard(matLab);
  //   }, []);
  const copyCodeToClipboard = (code) => {
    navigator.clipboard
      .writeText(code)
      .then(() => {
        console.log("Code copied to clipboard");
      })
      .catch((err) => {
        console.error("Unable to copy code to clipboard", err);
      });
  };

  return (
    <div className="App">
      <button onClick={() => copyCodeToClipboard(graphColoring)}>
        Graph Coloring
      </button>
      <button onClick={() => copyCodeToClipboard(xor)}>Xor </button>
      <button onClick={() => copyCodeToClipboard(kanpsack)}>Knapsack </button>
      <button onClick={() => copyCodeToClipboard(dfs)}>DFS </button>
      <button onClick={() => copyCodeToClipboard(water)}>Water</button>
      <button onClick={() => copyCodeToClipboard(token)}>Token</button>
      <button onClick={() => copyCodeToClipboard(aStart)}>A*</button>
      <button onClick={() => copyCodeToClipboard(singlePLayer)}>
        Single Player Game
      </button>
      <button onClick={() => copyCodeToClipboard(twoPlayer)}>
        2 Player Tic Tac
      </button>
    </div>
  );
}

export default App;
{
  /* <button onClick={() => copyCodeToClipboard(graphColoring)}>
        Graph Coloring
      </button>
      <button onClick={() => copyCodeToClipboard(xor)}>Xor </button>
      <button onClick={() => copyCodeToClipboard(kanpsack)}>Knapsack </button>
      <button onClick={() => copyCodeToClipboard(dfs)}>DFS </button>
      <button onClick={() => copyCodeToClipboard(water)}>Water</button>
      <button onClick={() => copyCodeToClipboard(token)}>Token</button>
      <button onClick={() => copyCodeToClipboard(matLab)}>MAT LAB</button>
      <button onClick={() => copyCodeToClipboard(matLab)}>MAT LAB</button> */
}
