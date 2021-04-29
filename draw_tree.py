from turtle import Turtle


def draw(yoshi: Turtle, length: int) -> None:
    """Draw a recursive tree pattern.
    """
    if length < 16:
        return

    yoshi.forward(length)
    yoshi.left(30)
    draw(yoshi, 3 * length // 4)
    yoshi.right(60)
    draw(yoshi, 3 * length // 4)
    yoshi.left(30)
    yoshi.backward(length)


if __name__ == '__main__':

    yoshi = Turtle()
    yoshi.right(90)
    yoshi.forward(128)
    yoshi.right(180)
    yoshi.speed(15)
    draw(yoshi, 128)

    win = yoshi.getscreen()
    win.exitonclick()
