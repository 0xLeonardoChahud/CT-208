import pytest
import Suguru.InstanceGenerator as InstanceGenerator


@pytest.mark.timeout(2)
def test_4x4():
    rows = 4
    cols = 4
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_5x5():
    rows = 5
    cols = 5
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_6x6():
    rows = 6
    cols = 6
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_7x7():
    rows = 7
    cols = 7
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_8x8():
    rows = 8
    cols = 8
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_9x9():
    rows = 9
    cols = 9
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_10x10():
    rows = 10
    cols = 10
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_11x11():
    rows = 11
    cols = 11
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_12x12():
    rows = 12
    cols = 12
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_13x13():
    rows = 13
    cols = 13
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_14x14():
    rows = 14
    cols = 14
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_15x15():
    rows = 15
    cols = 15
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_16x16():
    rows = 16
    cols = 16
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_17x17():
    rows = 17
    cols = 17
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_18x18():
    rows = 18
    cols = 18
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_19x19():
    rows = 19
    cols = 19
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()


@pytest.mark.timeout(2)
def test_20x20():
    rows = 20
    cols = 20
    game = InstanceGenerator.SuguruGenerator(rows, cols)
    for i in range(100):
        game.generate()
