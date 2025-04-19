# -------- Stage 1: Builder --------
FROM debian:bookworm-slim as builder

ENV PGDATA=/var/lib/postgresql/data
ENV POSTGRES_DB=datastore

# Install PostgreSQL, zstd, and dependencies
RUN apt-get update && \
    apt-get install -y postgresql-15 postgresql-client-15 zstd iputils-ping locales && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Set locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

USER postgres
RUN mkdir -p $PGDATA && chown -R postgres:postgres $PGDATA
RUN /usr/lib/postgresql/15/bin/initdb -D $PGDATA
COPY --chown=postgres:postgres postgres.conf $PGDATA/postgresql.conf

RUN echo "host all all 0.0.0.0/0 trust" >> $PGDATA/pg_hba.conf && \
    echo "local all all trust" >> $PGDATA/pg_hba.conf

# Initialize and import database
RUN --mount=type=bind,source=pg.zst,target=/tmp/pg.zst \ 
    /usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA start && \
    createdb --encoding=UTF8 --lc-collate='en_US.utf8' --lc-ctype='en_US.utf8' --template=template0 $POSTGRES_DB && \
    psql -d $POSTGRES_DB -c "CREATE USER chris WITH SUPERUSER PASSWORD 'chris';" && \
    psql -d $POSTGRES_DB -c "CREATE USER willbaldwin WITH SUPERUSER PASSWORD 'willbaldwin';" && \
    zstd -d --memory=$MEMORY_LIMIT /tmp/pg.zst -c | psql -d $POSTGRES_DB --set ON_ERROR_STOP=1 && \
    /usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA -m fast -w stop

# -------- Stage 2: Final Image --------
FROM debian:bookworm-slim

ENV PGDATA=/var/lib/postgresql/data
ENV POSTGRES_DB=datastore

# Install only what's needed to run Postgres
RUN apt-get update && \
    apt-get install -y postgresql-15 postgresql-client-15 iputils-ping locales && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Set locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Copy over the fully populated PGDATA dir from builder
COPY --from=builder --chown=postgres:postgres /var/lib/postgresql/data /var/lib/postgresql/data

USER postgres

# Expose Postgres port (optional)
EXPOSE 5432

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres -d $POSTGRES_DB

# Start PostgreSQL
CMD ["/usr/lib/postgresql/15/bin/postgres", "-D", "/var/lib/postgresql/data"]
