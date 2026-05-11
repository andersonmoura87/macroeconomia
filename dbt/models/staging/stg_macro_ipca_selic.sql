-- Lê Parquet gerado pelo pipeline Python (ajuste o caminho se necessário).
{{ config(materialized='view') }}

select *
from read_parquet('../../data/processed/macro_ipca_selic.parquet')
