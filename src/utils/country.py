import pycountry


def standardize_countries(df):
    def get_country_name(code):
        try:
            return pycountry.countries.lookup(code).name
        except LookupError:
            return code

    df["main_country"] = df["main_country_code"].apply(get_country_name)
    return df
