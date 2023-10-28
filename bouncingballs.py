import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 900, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Balls")

# Ball parameters
ball_radius = 30
balls = []

# Ball class
class Ball:
    def __init__(self, x, y, radius, color, x_speed, y_speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.wall_bounces = 0
        self.ball_bounces = 0

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

    def check_collision(self):
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.x_speed *= -1
            self.wall_bounces += 1
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.y_speed *= -1
            self.wall_bounces += 1

    def check_ball_collisions(self, other_balls):
        for other in other_balls:
            if self != other:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                if distance <= 2 * self.radius:
                    # Swap velocities to simulate bouncing off each other
                    self.x_speed, other.x_speed = other.x_speed, self.x_speed
                    self.y_speed, other.y_speed = other.y_speed, self.y_speed
                    self.ball_bounces += 1
                    other.ball_bounces += 1

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        font = pygame.font.Font(None, 36)
        text = font.render(str(self.wall_bounces + self.ball_bounces), True, (0, 0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (self.x, self.y)
        screen.blit(text, text_rect)

# Function to generate a random color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Create initial balls with random colors and slower speeds
for _ in range(6):
    x = random.randint(ball_radius, WIDTH - ball_radius)
    y = random.randint(ball_radius, HEIGHT - ball_radius)
    x_speed = random.uniform(-1.0, 1.0) * 1  # Adjust the factor for slower/faster balls
    y_speed = random.uniform(-1.0, 1.0) * 1  # Adjust the factor for slower/faster balls
    color = random_color()
    ball = Ball(x, y, ball_radius, color, x_speed, y_speed)
    balls.append(ball)

# Main game loop
running = True
message_displayed = False  # To track if "Message" has been displayed
message_display_time = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Fill the screen with a black background

    for ball in balls:
        ball.move()
        ball.check_collision()
        ball.check_ball_collisions(balls)
        ball.draw()

        if not message_displayed and ball.wall_bounces + ball.ball_bounces >= 15:
            message_font = pygame.font.Font(None, 150)
            message_text = message_font.render(" Buen Aniversario Amor ", True, (255, 255, 255))
            message_rect = message_text.get_rect()
            message_rect.center = (WIDTH // 2, HEIGHT // 2)
            screen.fill((0, 0, 0))  # Clear the screen
            screen.blit(message_text, message_rect)
            pygame.display.update()
            message_displayed = True
            message_display_time = pygame.time.get_ticks()
            pygame.time.delay(50000)  # Pause for 10 seconds

        if message_displayed and pygame.time.get_ticks() - message_display_time >= 10000:
            # Display "Message" for 10 seconds
            running = False

    pygame.display.update()

# Wait for the user to close the game
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
