

# allow for drawing an infinite number of batches from the dataloader
def loop_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch