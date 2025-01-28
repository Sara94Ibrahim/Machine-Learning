#!/usr/bin/env python
# coding: utf-8

# # Cancer Cell microarray data example

# ## The NCI60 data



nci_labs = pd.read_csv("nci60_labs.csv", index_col = 0)
nci_data = pd.read_csv("nci60_data.csv", index_col = 0)




print(nci_data.shape) #64 rows, 6830 columns



print(nci_labs.x.value_counts(sort = True))



nci_data.index = nci_labs.x

fig, ax = plt.subplots(3,1, figsize=(15,22))
fig.subplots_adjust(hspace=0.5)

linkages = ['complete', 'single', 'average']
for link, axis in zip(linkages, fig.axes):
    hc = linkage(y = nci_data, method = link, metric = 'euclidean')
    axis.set_title("Linkage=%s" % link, size =15)
    axis.set_xlabel('Sample Type', size = 15)
    axis.set_ylabel('Distance', size = 15)
    dendrogram(hc, ax=axis, labels=nci_data.index, leaf_rotation = 90, leaf_font_size =10)
    
plt.show()



nci_hc_complete = linkage( y = nci_data, method = 'complete', metric = 'euclidean')
nci_hc_complete_4_clusters = cut_tree(nci_hc_complete, n_clusters = 4)
print(pd.crosstab(index = nci_data.index,
                  columns = nci_hc_complete_4_clusters.T[0],
                  rownames = ['Cancer Type'],
                  colnames = ['Cluster']))



fig, ax = plt.subplots(1,1, figsize = (15,8))
dendrogram(nci_hc_complete,
           labels = nci_data.index,
           leaf_font_size = 14,
           show_leaf_counts = True)
plt.axhline(y = 110, c = 'k', ls = 'dashed')
plt.show()



kmean_4 = KMeans(n_clusters = 4, random_state = 123, n_init = 150)
kmean_4.fit(nci_data)
print(kmean_4.labels_)


print(pd.crosstab(index = kmean_4.labels_,
                  columns = nci_hc_complete_4_clusters.T[0],
                  rownames = ['K-Means'],
                  colnames = ['Hierarchical']))
      

