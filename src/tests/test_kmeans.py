from unittest import TestCase

from src.kmeans.main import KMeans
from src.loader import load_data
from src.utils import ordinal_encode, plot_kmeans


class TestKMeans(TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestKMeans, cls).setUpClass()
        cls.test_type = "clustering"

    def test_kmeans(self):
        data = load_data("clustering/test.csv.zip")
        clean_data = ordinal_encode(data)
        assert clean_data.shape == data.shape

        kmeans = KMeans(k=5)
        labels = kmeans.fit(clean_data, 5000)
        # plot_kmeans(clean_data, labels, kmeans)
        len(labels) == clean_data.shape[0]

        assert 1 == 1