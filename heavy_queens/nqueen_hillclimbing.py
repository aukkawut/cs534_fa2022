import numpy as np
np.random.seed(8)
import time
start_time = time.time()
def heuristic(board):
    '''
    This function will calculate the heuristic value of the given board
    '''
    def diagonal_conflict(chessboard):
        '''This function count the number of diagonal conflicts in the chessboard'''
        count = 0
        for i in range(len(chessboard)):
            for j in range(len(chessboard)):
                if chessboard[i][j] != 0:
                    for k in range(len(chessboard)):
                        for l in range(len(chessboard)):
                            if chessboard[k][l] != 0:
                                if abs(i-k) == abs(j-l) and (i != k and j != l):
                                    count += 1
        return count
    def vertical_conflict(chessboard):
        '''This function count the number of vertical conflicts in the chessboard'''
        count = 0
        for i in range(len(chessboard)):
            for j in range(len(chessboard)):
                if chessboard[i][j] != 0:
                    for k in range(len(chessboard)):
                        for l in range(len(chessboard)):
                            if chessboard[k][l] != 0:
                                if i == k and j != l:
                                    count += 1
        return count
    def horizontal_conflict(chessboard):
        '''This function count the number of horizontal conflicts in the chessboard'''
        count = 0
        for i in range(len(chessboard)):
            for j in range(len(chessboard)):
                if chessboard[i][j] != 0:
                    for k in range(len(chessboard)):
                        for l in range(len(chessboard)):
                            if chessboard[k][l] != 0:
                                if j == l and i != k:
                                    count += 1
        return count
    return diagonal_conflict(board) + vertical_conflict(board) + horizontal_conflict(board)
def heuristic_matrix(board):
    '''
    This function will generate the heuristic matrix of a given board
    h_ij means that if the queen at the row i or column j were to move there,
    what would be the heuristic
    '''
    h = 1000*np.ones((len(board), len(board))) #penalize the queen position 
    queens = [list(i) for i in zip(np.where(board > 0)[0], np.where(board > 0)[1])]
    for i in range(len(board)):
        for j in range(len(board)):
            #check if the queen is already there
            if board[i][j] == 0:
                board_buf = np.copy(board)
                hl=[]
                for x in queens:
                    #move queen at that row (i) to column j
                    if x[0] == i:
                        board_buf[x[0]][x[1]] = 0
                        board_buf[i][j] = 1
                        hl.append(heuristic(board_buf))
                        #print(f"move {x} to {i, j} and heuristic is {heuristic(board_buf)}")
                        board_buf[i][j] = 0
                        board_buf[x[0]][x[1]] = 1
                    elif x[1] == j:
                        board_buf[x[0]][x[1]] = 0
                        board_buf[i][j] = 1
                        hl.append(heuristic(board_buf))
                        #print(f"move {x} to {i, j} and heuristic is {heuristic(board_buf)}")
                        board_buf[i][j] = 0
                        board_buf[x[0]][x[1]] = 1
                    try:
                        h[i][j] = min(hl)
                    except:
                        pass
    return h
def hillclimbing(board,max_iter=1000, max_stuck=20):
    '''
    This function will run the hill climbing algorithm to find the solution of n-queen
    of a given board
    '''
    count_stuck = 0
    count = 0
    board_buf = np.copy(board)
    hlist = []
    moves = []
    while(True):
        if heuristic(board_buf) == 0 or count_stuck >= max_stuck or count >= max_iter:
            if count_stuck >= max_stuck:
                print(f"The algorithm did not converge in {max_stuck} iterations")
                print(f"Performing random restart")
                np.random.seed(np.random.randint(0,1000))
                count_stuck = 0
                hlist = []
                moves = []
                board_buf = np.copy(board)
            elif count >= max_iter:
                print(f"The algorithm did not converge in {max_iter} iterations")
                break
            else:
                print("The algorithm converged")
                print(board_buf)
                break
            #return board
        count+=1
        h = heuristic_matrix(board_buf)
        hlist.append(heuristic(board_buf))
        if min(hlist) == hlist[-1]:
            count_stuck += 1
        movable_pos = [list(i) for i in zip(np.where(h == np.min(h))[0], np.where(h == np.min(h))[1])]
        moving_pos = movable_pos[np.random.choice(len(movable_pos),1)[0]]
        #print(movable_pos)
        #print(moving_pos)
        #choose the queen on that row or column uniformly at random
        try:
            queens_row = [[moving_pos[0],x[0]] for x in np.where(board_buf[moving_pos[0],:] > 0)]
        except:
            queens_row = []
        try:
            queens_column = [[x[0],moving_pos[1]] for x in np.where(board_buf[:,moving_pos[1]] > 0)]
        except:
            queens_column = []
        queens = queens_row + queens_column
        if len(queens) > 0:
            moving_queen = queens[np.random.choice(len(queens),1)[0]]
        else:
            print("No queen to move")
            print(board)
            print(h)
            print(np.where(h == np.min(h)))
            raise Exception("No queen to move")
        #print(moving_queen)
        #move the queen to the position
        print(f"move {moving_queen} to {moving_pos} and heuristic is {heuristic(board_buf)}")
        moves.append([moving_queen, moving_pos])
        k = board_buf[moving_queen[0]][moving_queen[1]]
        board_buf[moving_queen[0]][moving_queen[1]] = 0
        board_buf[moving_pos[0]][moving_pos[1]] = k
        print(board_buf)
        #print(h)
        if len(np.where(board_buf > 0)[0]) > len(board):
            raise Exception("Extra queen found")
    return moves
def main():
    '''
    This is the main function
    '''
    #generate a random board
    n = 10
    board = np.zeros((n, n))
    board[np.random.choice(n, n, replace=False), np.arange(n)] = np.random.randint(1, 9,n)
    print("The initial board is:")
    print(board)
    print("The heuristic of the initial board is:")
    print(heuristic(board))
    print("The heuristic matrix of the initial board is:")
    print(heuristic_matrix(board))
    #decision    
    moves = hillclimbing(board)
    for x in moves:
        print(f"move {x[0]} to {x[1]}")
    #moving_queen = 
    #print(moving_queen)
    #print(hillclimbing(board))

if __name__ == '__main__':
    main()
    print(f"runtime: {time.time() - start_time} s" )