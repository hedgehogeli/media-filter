# Overview:

A messy POC project meant to learn something while (attempting to) emulate the big ML training schemes + working within hardware limitations; and also get a useful result out of it.

CLIP + YOLO backbone image classifier meant to classify media content (image and video), without any text context (left as potential for extension) into 3 classes: BAD, NEUTRAL, GOOD

deployed as a scraper client + backend server running classifier model. 

client (running in a VM) collects image data, compresses, and sends to the backend. 

backend server takes a job queue and does image preprocessing in parallel, then batches images for inference. I would put this on my actual server if I had the hardware for it, but no GPU on my server.

lots of struggles with hardware limitations on this one, moved from windows -> WSL -> linux (thank you windows 11 /s) for parallelisation

# file structure

repo is an agglomeration of 2 repos I usually keep on my own upstream server, 

`usage` is basically my deployment code, `classifier` is the model training stuff 

`classifier/model/iteration4.ipynb` is the latest version of training notebook

`usage/deployment` contains the client/server protocol code


# Training data 
is 2M images (~200GB)
- basically just my viewing/like history across platforms for GOOD class, 
- BAD dataset is primarily cobbled together by searching for keywords (e.g. all of Mr Beast's YT thumbnails, Reddit bot reposters)
- NEUTRAL dataset is just raw unlabelled data, which by my estimate is probably 30% BAD and 5% GOOD by composition

Scraping is done through Python Playwright, and parallelised with Python's multiprocessing

Image data is resized, compressed to be under 200KB, reformatted (FFmpeg, all in RAM to save my poor SSD)

Video data is reduced to a thumbnail, and a few screenshots of the video at arbitrary durations (latter is a suspicious choice)

# deployment

no docker or anything, didn't want to bother with the overhead, just running the python was sufficient for me

# Results

Deployed this for around a month before Cloudflare broke my scrapers and I decided it was a more efficient use of my time to stop consuming content I need to build a filter for, and just altogether stopped... 

Had around 50% recall on the bad class (really the only class of importance), which means that I had to see around 50% less "garbage", which is just a huge win in my books.

Very poor performance around the GOOD class, which I attribute to poor data, but I didn't have high hopes here anyway.


# Approach

clip for object recognition, yolo face models to hopefully gain performance on clickbait thumbnails

asymmetric focal loss to try to compensate for class imbalance, and to penalise egregious classification (true good classified as bad)

mean teacher chosen as a simple route into unsupervised learning on unlabelled data; suitable for dirty labels

transforms are engineered around the fact that rotate+resize is incredibly expensive on CPU

warmup and frozen weights to ramp up training and avoid instability

teacher confidence weighting -> apply more consistency loss to student when teacher is confident

