for i in range(1,16):
    for item in ['bow','dolphin']:
        for sentiment in ['pos','neg']:
            f = open(f'data/reviews/{item}/{sentiment}/{i}.txt','w')
            f.close()