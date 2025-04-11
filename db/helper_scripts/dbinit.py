def createTables(conn):
    cur = conn.cursor()

    cur.execute("""
    DO $$ 
    BEGIN
        EXECUTE (
            SELECT string_agg('DROP TABLE IF EXISTS ' || table_name || ' CASCADE;', ' ')
            FROM information_schema.tables
            WHERE table_schema = 'public'
        );
    END $$;
    """)

    with open("device_event.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("device.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("recallenforcement.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("classification.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("patient.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("mdr_text.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("510k.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("premarketapproval.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    with open("recalls.sql", "r") as f:
        sql = f.read()
    cur.execute(sql)
    conn.commit()

    cur.close()



