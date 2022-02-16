import imageio

filenames = [f"/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Miscellaneous/jpgs/IMG_{i}.jpeg" for i in range(2799, 2804)]
movie_path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Miscellaneous/jpgs/movie.gif"

with imageio.get_writer(movie_path, mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)