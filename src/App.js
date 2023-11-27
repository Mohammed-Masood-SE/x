import React from "react";
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
  class Node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

class Tree:
    def __init__(self):
        self.head = None

    def insert(self,value):
        newNode = Node(value)
        if self.head == None:
            self.head = newNode
            return
        current = self.head
        while current:
            if value < current.value:
                if current.left == None:
                    current.left = newNode
                    return
                current = current.left
            else:
                if current.right == None:
                    current.right = newNode
                    return
                current = current.right
    
    def DFS_InOrder(self):
        data = []
        current = self.head
        def traverse(node):
            
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
            
        traverse(current)
        return data

            

        


#   10
#  5   12
# 4  6 11 13
tree = Tree()
tree.insert(10)
tree.insert(5)
tree.insert(4)
tree.insert(12)
tree.insert(11)
tree.insert(13)
tree.insert(6)
print(tree.DFS_InOrder())

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
    </div>
  );
}

export default App;
