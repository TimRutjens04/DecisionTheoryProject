import pygame
from pygame import mixer
import sys
from pathlib import Path

pygame.init()
mixer.init()

#Load sfx
music_path = Path(__file__).parent / "assets" / "sfx" / "checkmate.mp3"
mixer.music.load(music_path)

WINDOW_SIZE = 600
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
LIGHT_SQUARES = (238, 238, 210)
DARK_SQUARES = (118, 150, 86)
PIECE_SIZE = WINDOW_SIZE // 5

pygame.display.set_caption("Chess Board")

def draw_start_button():
    """Draw and return a start button in the UI"""
    button_width = 200
    button_height = 50
    button_x = (WINDOW_SIZE - button_width) // 2
    button_y = WINDOW_SIZE // 2
    
    button = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(SCREEN, (34, 139, 34), button)
    
    font = pygame.font.Font(None, 36)
    text = font.render("Start Match", True, (255, 255, 255))
    text_rect = text.get_rect(center=button.center)
    SCREEN.blit(text, text_rect)
    
    return button

def load_pieces():
    pieces = {}
    piece_path = Path(__file__).parent / "assets" / "pieces"
    for color in ['white', 'black']:
        for piece in ['king', 'rook', 'bishop']:
            image = pygame.image.load(piece_path / f"{color}_{piece}.png")
            image = pygame.transform.scale(image, (PIECE_SIZE, PIECE_SIZE))
            pieces[f"{color}_{piece}"] = image
    return pieces

PIECES = load_pieces()

def draw_piece(piece_name, x, y):
    if piece_name in PIECES:
        SCREEN.blit(PIECES[piece_name], (x, y))

def drawGrid(game_state=None):
    global WINDOW_SIZE
    # Grid/Board
    blockSize = WINDOW_SIZE // 5
    for row in range(5):
        for column in range(5):
            x = column * blockSize
            y = row * blockSize
            square = pygame.Rect(x, y, blockSize, blockSize)
            color = LIGHT_SQUARES if (row + column) % 2 == 0 else DARK_SQUARES
            pygame.draw.rect(SCREEN, color, square)

    # Pieces
    board_layout = game_state.board if game_state else [
        ['.', 'bR', 'bK', 'bB', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', 'wR', 'wK', 'wB', '.']
    ]

    for row in range(5):
        for column in range(5):
            piece = board_layout[row][column]
            if piece != '.':
                color = 'white' if piece[0] == 'w' else 'black'
                piece_type = piece[1].lower()
                match piece_type:
                    case 'k':
                        piece_type = 'king'
                    case 'r':
                        piece_type = 'rook'
                    case 'b':
                        piece_type = 'bishop'
                piece_name = f"{color}_{piece_type}"
                x = column * PIECE_SIZE
                y = row * PIECE_SIZE
                draw_piece(piece_name, x, y)

def signal_game_end(winner):
    #SFX Logic
    mixer.music.set_volume(0.7)
    mixer.music.play()

    #Display banner with winner
    button_width = 200
    button_height = 50
    button_x = (WINDOW_SIZE - button_width) // 2
    button_y = WINDOW_SIZE // 2
    
    button = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(SCREEN, (0, 0, 0), button)
    
    font = pygame.font.Font(None, 36)
    if winner == 'draw':
        text = font.render("Game ended in a draw", True, (255, 255, 255))
    else:
        text = font.render(f"Winner: {winner}", True, (255, 255, 255))
    text_rect = text.get_rect(center=button.center)
    SCREEN.blit(text, text_rect)
    
    return button

def main():
    global SCREEN
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False               
                pygame.quit()
                sys.exit()

        drawGrid()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()

    