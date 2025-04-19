# Database Setup Instructions

We are leveraging postgresql 15 since it's widely available on LTS operating systems. Easier for the TAs to get for sure. We recommend 300GB of free space to start with

### Create Database
```
initdb -D [location]
```

#### Start Database
```
pg_ctl -D [location] -l logfile start
```

**NOTE**: *You may need to change the following line to match in postgresql.conf*

Remember to uncomment the line as well:
```
unix_socket_directories = '/tmp'
```

### db export
```
pg_dump -U [user] -d [database] -h [host] | zstd -z -13 -o pg.zst
```

### db import
```
zstd --stdout -d pg.zst | psql --dbname=[name] -U [user]
```


### Dockerfile.big 
This is for building the container with the full-size database. This makes the local build bigger but the resulting container smaller due to skipping a file copy