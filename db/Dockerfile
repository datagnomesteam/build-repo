FROM debian:bookworm-slim

# Install PostgreSQL 15, zstd, aria2, and other tools
RUN apt-get update && \
    apt-get install -y postgresql-15 postgresql-client-15 zstd pv && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PGDATA=/var/lib/postgresql/data
ENV POSTGRES_DB=datastore

# Create database directory
RUN mkdir -p $PGDATA && chown -R postgres:postgres $PGDATA

# Copy or download dump
COPY ./pg.zst /tmp/pg.zst
RUN chmod -R 777 /tmp

# Use postgres user
USER postgres

# Initialize PostgreSQL
RUN /usr/lib/postgresql/15/bin/initdb -D $PGDATA

RUN echo "host all all 0.0.0.0/0 trust" >> $PGDATA/pg_hba.conf && \
    echo "local all all trust" >> $PGDATA/pg_hba.conf

# Start temporary server, import data in chunks, then stop server
RUN /usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA start && \
    createdb $POSTGRES_DB && \
    psql -d $POSTGRES_DB -c "CREATE USER chris WITH SUPERUSER PASSWORD 'chris';" && \
    psql -d $POSTGRES_DB -c "CREATE USER willbaldwin WITH SUPERUSER PASSWORD 'willbaldwin';" && \
    zstd -d --memory=$MEMORY_LIMIT /tmp/pg.zst -c | pv -s $(zstd -l /tmp/pg.zst | grep "Decompressed" | awk '{print $3}') | psql --set ON_ERROR_STOP=1 && \
    /usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA -m fast -w stop && rm /tmp/pg.zst

# Expose Postgres port (optional)
EXPOSE 5432

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres -d $POSTGRES_DB

# Default command
CMD ["/usr/lib/postgresql/15/bin/postgres", "-D", "/var/lib/postgresql/data"]