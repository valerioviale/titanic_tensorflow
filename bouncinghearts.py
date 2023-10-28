# Import necessary libraries
import pygame  # For game development
import random  # For generating random values
import math    # For mathematical calculations

# Initialize Pygame
pygame.init()

# Define the screen dimensions
WIDTH, HEIGHT = 1250, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Balls")  # Set the window title

# Parameters for the bouncing balls
ball_radius = 25
balls = []  # List to store ball objects

# Define the Ball class to represent bouncing balls
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

    # Move the ball by updating its position
    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

    # Check for collisions with the walls and adjust velocity
    def check_collision(self):
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.x_speed *= -1
            self.wall_bounces += 1
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.y_speed *= -1
            self.wall_bounces += 1

    # Check for collisions with other balls and swap velocities
    def check_ball_collisions(self, other_balls):
        for other in other_balls:
            if self != other:
                distance = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
                if distance <= 2 * self.radius:
                    # Swap velocities to simulate bouncing off each other
                    self.x_speed, other.x_speed = other.x_speed, self.x_speed
                    self.y_speed, other.y_speed = other.y_speed, self.y_speed
                    self.ball_bounces += 1
                    other.ball_bounces += 1

    # Draw the ball on the screen and display bounce counts
    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        font = pygame.font.Font(None, 32)
        text = font.render(str(self.wall_bounces + self.ball_bounces), True, (0, 0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (self.x, self.y)
        screen.blit(text, text_rect)

# Function to generate a random RGB color
def random_light_color():
    # Generate random values within a range for lighter colors
    red = random.randint(100, 255)
    green = random.randint(200, 255)
    blue = random.randint(0, 255)
    return (red, green, blue)

# Create initial balls with random light colors and slower speeds
for _ in range(6):
    x = random.randint(ball_radius, WIDTH - ball_radius)
    y = random.randint(ball_radius, HEIGHT - ball_radius)
    x_speed = random.uniform(-1.0, 1.0) * 1
    y_speed = random.uniform(-1.0, 1.0) * 1
    color = random_light_color()  # Use the new function for lighter colors
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

    # Fill the screen with a black background
    screen.fill((0, 0, 0))

    # Iterate through the list of balls
    for ball in balls:
        ball.move()  # Move the ball
        ball.check_collision()  # Check for wall collisions
        ball.check_ball_collisions(balls)  # Check for collisions with other balls
        ball.draw()  # Draw the ball on the screen

        # Display "Message" if ball bounces exceed a threshold and wait for a set time
        if not message_displayed and ball.wall_bounces + ball.ball_bounces >= 15:
            message_font = pygame.font.Font(None, 150)
            message_text = message_font.render(" Message ", True, (255, 255, 255))
            message_rect = message_text.get_rect()
            message_rect.center = (WIDTH // 2, HEIGHT // 2)
            screen.fill((0, 0, 0))  # Clear the screen
            screen.blit(message_text, message_rect)
            pygame.display.update()
            message_displayed = True
            message_display_time = pygame.time.get_ticks()
            pygame.time.delay(10000)  # Pause for 10 seconds

        # Exit the game after displaying "Message" for a set time
        if message_displayed and pygame.time.get_ticks() - message_display_time >= 10000:
            running = False

    pygame.display.update()

# Wait for the user to close the game
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
