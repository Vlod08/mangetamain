# bubble_map.py
from __future__ import annotations
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from branca.element import Template, MacroElement

# import site for geopandas / geodatasets if available
try:
    import geopandas as gpd  # one place only
except Exception:
    gpd = None

try:
    import geodatasets as gds
except Exception:
    gds = None


class BubbleMapFolium:
    """
    Folium bubble map where radius reflects recipes count.
    - Accepts data grouped by 'country' or 'continent' with a 'count_recipes' column.
    - If auto_centroids=True (default), computes centroids via GeoPandas/Natural Earth.
    Otherwise uses provided 'country_centroids' or a minimal built-in table.
    """

    # Minimal fallback if GeoPandas is unavailable
    _COUNTRY_CENTROIDS_POOL = pd.DataFrame(
        [
    # Europe
    ("France", 48.8566, 2.3522),
    ("Germany", 52.5200, 13.4050),
    ("Spain", 40.4168, -3.7038),
    ("Italy", 41.9028, 12.4964),
    ("United Kingdom", 51.5074, -0.1278),
    ("Portugal", 38.7223, -9.1393),
    ("Netherlands", 52.3676, 4.9041),
    ("Belgium", 50.8503, 4.3517),
    ("Luxembourg", 49.6117, 6.1319),
    ("Ireland", 53.3498, -6.2603),
    ("Switzerland", 46.9480, 7.4474),
    ("Austria", 48.2082, 16.3738),
    ("Poland", 52.2297, 21.0122),
    ("Czechia", 50.0755, 14.4378),
    ("Slovakia", 48.1486, 17.1077),
    ("Hungary", 47.4979, 19.0402),
    ("Slovenia", 46.0569, 14.5058),
    ("Croatia", 45.8150, 15.9819),
    ("Bosnia and Herzegovina", 43.8563, 18.4131),
    ("Serbia", 44.7866, 20.4489),
    ("Montenegro", 42.4304, 19.2594),
    ("North Macedonia", 41.9981, 21.4254),
    ("Albania", 41.3275, 19.8187),
    ("Greece", 37.9838, 23.7275),
    ("Bulgaria", 42.6977, 23.3219),
    ("Romania", 44.4268, 26.1025),
    ("Moldova", 47.0105, 28.8638),
    ("Ukraine", 50.4501, 30.5234),
    ("Belarus", 53.9006, 27.5590),
    ("Lithuania", 54.6872, 25.2797),
    ("Latvia", 56.9496, 24.1052),
    ("Estonia", 59.4370, 24.7536),
    ("Norway", 59.9139, 10.7522),
    ("Sweden", 59.3293, 18.0686),
    ("Finland", 60.1699, 24.9384),
    ("Denmark", 55.6761, 12.5683),
    ("Iceland", 64.1466, -21.9426),
    ("Andorra", 42.5078, 1.5211),
    ("Monaco", 43.7384, 7.4246),
    ("Liechtenstein", 47.1410, 9.5209),
    ("San Marino", 43.9356, 12.4473),
    ("Vatican City", 41.9029, 12.4534),
    ("Turkey", 39.9208, 32.8541),

    # North America
    ("United States", 38.9072, -77.0369),
    ("Canada", 45.4215, -75.6972),
    ("Mexico", 19.4326, -99.1332),
    ("Guatemala", 14.6349, -90.5069),
    ("Belize", 17.5046, -88.1962),
    ("Honduras", 14.0723, -87.1921),
    ("El Salvador", 13.6929, -89.2182),
    ("Nicaragua", 12.1364, -86.2514),
    ("Costa Rica", 9.9281, -84.0907),
    ("Panama", 8.9824, -79.5199),
    ("Cuba", 23.1136, -82.3666),
    ("Jamaica", 18.0179, -76.8099),
    ("Haiti", 18.5944, -72.3074),
    ("Dominican Republic", 18.4861, -69.9312),
    ("Bahamas", 25.0343, -77.3963),
    ("Barbados", 13.0975, -59.6145),
    ("Trinidad and Tobago", 10.6667, -61.5167),

    # South America
    ("Brazil", -15.7939, -47.8828),
    ("Argentina", -34.6037, -58.3816),
    ("Chile", -33.4489, -70.6693),
    ("Uruguay", -34.9011, -56.1645),
    ("Paraguay", -25.2637, -57.5759),
    ("Bolivia", -16.5000, -68.1500),
    ("Peru", -12.0464, -77.0428),
    ("Ecuador", -0.1807, -78.4678),
    ("Colombia", 4.7110, -74.0721),
    ("Venezuela", 10.4910, -66.9026),
    ("Guyana", 6.8013, -58.1553),
    ("Suriname", 5.8520, -55.2038),

    # Africa
    ("Morocco", 34.0209, -6.8416),
    ("Algeria", 36.7538, 3.0588),
    ("Tunisia", 36.8065, 10.1815),
    ("Libya", 32.8872, 13.1913),
    ("Egypt", 30.0444, 31.2357),
    ("Mauritania", 18.0790, -15.9650),
    ("Mali", 12.6392, -8.0029),
    ("Niger", 13.5116, 2.1254),
    ("Chad", 12.1348, 15.0557),
    ("Sudan", 15.5007, 32.5599),
    ("South Sudan", 4.8517, 31.5825),
    ("Eritrea", 15.3229, 38.9251),
    ("Ethiopia", 9.0108, 38.7613),
    ("Djibouti", 11.5721, 43.1456),
    ("Somalia", 2.0469, 45.3182),
    ("Senegal", 14.7167, -17.4677),
    ("Gambia", 13.4549, -16.5790),
    ("Guinea", 9.6412, -13.5784),
    ("Guinea-Bissau", 11.8636, -15.5820),
    ("Sierra Leone", 8.4657, -13.2317),
    ("Liberia", 6.3156, -10.8074),
    ("Côte d'Ivoire", 5.3600, -4.0083),
    ("Ghana", 5.6037, -0.1870),
    ("Togo", 6.1319, 1.2228),
    ("Benin", 6.3703, 2.3912),
    ("Burkina Faso", 12.3714, -1.5197),
    ("Nigeria", 9.0765, 7.3986),
    ("Cameroon", 3.8480, 11.5021),
    ("Central African Republic", 4.3947, 18.5582),
    ("Equatorial Guinea", 3.7500, 8.7833),
    ("Gabon", 0.4162, 9.4673),
    ("Republic of the Congo", -4.2634, 15.2429),
    ("Democratic Republic of the Congo", -4.4419, 15.2663),
    ("Angola", -8.8390, 13.2894),
    ("Namibia", -22.5609, 17.0658),
    ("Botswana", -24.6282, 25.9231),
    ("Zimbabwe", -17.8292, 31.0522),
    ("Zambia", -15.3875, 28.3228),
    ("Malawi", -13.9626, 33.7741),
    ("Mozambique", -25.9667, 32.5833),
    ("Madagascar", -18.8792, 47.5079),
    ("Seychelles", -4.6796, 55.4920),
    ("Mauritius", -20.1609, 57.5012),
    ("Comoros", -11.6987, 43.2540),
    ("Cape Verde", 14.9177, -23.5092),
    ("Sao Tome and Principe", 0.3365, 6.7273),
    ("Rwanda", -1.9441, 30.0619),
    ("Burundi", -3.3614, 29.3599),
    ("Uganda", 0.3476, 32.5825),
    ("Kenya", -1.2921, 36.8219),
    ("Tanzania", -6.7924, 39.2083),
    ("Eswatini", -26.3054, 31.1367),
    ("Lesotho", -29.3167, 27.4833),
    ("South Africa", -25.7479, 28.2293),

    # Asia
    ("Russia", 55.7558, 37.6176),
    ("Kazakhstan", 51.1280, 71.4300),
    ("Uzbekistan", 41.2995, 69.2401),
    ("Turkmenistan", 37.9601, 58.3261),
    ("Kyrgyzstan", 42.8746, 74.5698),
    ("Tajikistan", 38.5598, 68.7870),
    ("Mongolia", 47.8864, 106.9057),
    ("China", 39.9042, 116.4074),
    ("Japan", 35.6762, 139.6503),
    ("South Korea", 37.5665, 126.9780),
    ("North Korea", 39.0392, 125.7625),
    ("Taiwan", 25.0330, 121.5654),
    ("Hong Kong", 22.3193, 114.1694),
    ("Macau", 22.1987, 113.5439),
    ("India", 28.6139, 77.2090),
    ("Pakistan", 33.6844, 73.0479),
    ("Afghanistan", 34.5553, 69.2075),
    ("Nepal", 27.7172, 85.3240),
    ("Bhutan", 27.4728, 89.6390),
    ("Bangladesh", 23.8103, 90.4125),
    ("Sri Lanka", 6.9271, 79.8612),
    ("Maldives", 4.1755, 73.5093),
    ("Myanmar", 19.7633, 96.0785),
    ("Thailand", 13.7563, 100.5018),
    ("Laos", 17.9757, 102.6331),
    ("Cambodia", 11.5564, 104.9282),
    ("Vietnam", 21.0278, 105.8342),
    ("Malaysia", 3.1390, 101.6869),
    ("Singapore", 1.3521, 103.8198),
    ("Indonesia", -6.2088, 106.8456),
    ("Philippines", 14.5995, 120.9842),
    ("Brunei", 4.9031, 114.9398),
    ("Timor-Leste", -8.5569, 125.5603),
    ("Iran", 35.6892, 51.3890),
    ("Iraq", 33.3152, 44.3661),
    ("Syria", 33.5138, 36.2765),
    ("Jordan", 31.9539, 35.9106),
    ("Lebanon", 33.8938, 35.5018),
    ("Israel", 31.7683, 35.2137),
    ("Palestine", 31.9522, 35.2332),
    ("Saudi Arabia", 24.7136, 46.6753),
    ("United Arab Emirates", 24.4539, 54.3773),
    ("Qatar", 25.2854, 51.5310),
    ("Bahrain", 26.2235, 50.5876),
    ("Kuwait", 29.3759, 47.9774),
    ("Oman", 23.5859, 58.4059),
    ("Yemen", 15.3694, 44.1910),

    # Oceania
    ("Australia", -35.2809, 149.1300),
    ("New Zealand", -41.2866, 174.7756),
    ("Papua New Guinea", -9.4780, 147.1500),
    ("Fiji", -18.1248, 178.4501),
    ("Solomon Islands", -9.4280, 159.9490),
    ("Vanuatu", -17.7333, 168.3273),
    ("Samoa", -13.8333, -171.7500),
    ("Tonga", -21.1394, -175.2049),
    ("Kiribati", 1.4518, 172.9717),
    ("Marshall Islands", 7.0897, 171.3803),
    ("Micronesia", 6.9147, 158.1610),
    ("Nauru", -0.5477, 166.9209),
    ("Tuvalu", -8.5201, 179.1983),
], 
columns=["country", "lat", "lon"])

    # init
    def __init__(
        self,
        tiles: str = "cartodbpositron",
        zoom_start: int = 2,
        control_scale: bool = True,
        country_centroids: pd.DataFrame | None = None,  # columns: ['country','lat','lon']
        auto_centroids: bool = True,                    # compute via GeoPandas if available
    ):
        self.tiles = tiles
        self.zoom_start = zoom_start
        self.control_scale = control_scale

        self._country_centroids = None       # DataFrame ['country','lat','lon']
        self._continent_centroids = None     # DataFrame ['continent','lat','lon']
        self._world_gdf = None               # cached GeoDataFrame for polygons

        # Highest priority: user-supplied centroids
        if country_centroids is not None:
            self._country_centroids = (
                country_centroids.rename(columns=str.lower)[["country", "lat", "lon"]].copy()
            )
            self._ensure_float_latlon(self._country_centroids)

        # If not provided, try to auto-build with GeoPandas (if requested)
        self.auto_centroids = auto_centroids
        if self._country_centroids is None and auto_centroids:
            self._try_autobuild_centroids()

        # If still none, fall back to minimal built-in table
        if self._country_centroids is None:
            self._country_centroids = self._COUNTRY_CENTROIDS_POOL.copy()

    # static data processing helpers
    @staticmethod
    def counts_by_country(df_raw: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
        return (
            df_raw.groupby(country_col, dropna=False)
            .size()
            .reset_index(name="count_recipes")
            .rename(columns={country_col: "country"})
        )

    @staticmethod
    def counts_by_continent(df_raw: pd.DataFrame, continent_col: str = "continent") -> pd.DataFrame:
        return (
            df_raw.groupby(continent_col, dropna=False)
            .size()
            .reset_index(name="count_recipes")
            .rename(columns={continent_col: "continent"})
        )
    
    # ensure lat/lon are floats
    @staticmethod
    def _ensure_float_latlon(df: pd.DataFrame):
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    @staticmethod
    def _scale_sizes(counts: np.ndarray, min_px: float, max_px: float, use_sqrt: bool = True) -> np.ndarray:
        x = np.sqrt(counts) if use_sqrt else counts.astype(float)
        if len(x) == 0 or np.isclose(x.max(), x.min()):
            return np.full_like(x, (min_px + max_px) / 2.0, dtype=float)
        x = (x - x.min()) / (x.max() - x.min())
        return min_px + x * (max_px - min_px)
    
    # color segments
    @staticmethod
    def _seg_color(v: int) -> str:
        """<10: red, 10–39: yellow, 40–60: blue, >60: green"""
        if v < 5:
            return "#e03131"  # red
        if v < 15:
            return "#f59f00"  # yellow
        if v <= 30:
            return "#1c7ed6"  # blue
        return "#2f9e44"      # green

    @staticmethod
    def _lighten(hex_color: str, factor: float = 0.55) -> str:
        """Blend color towards white by 'factor' (0..1)."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        lr = int(r + (255 - r) * factor)
        lg = int(g + (255 - g) * factor)
        lb = int(b + (255 - b) * factor)
        return f"#{lr:02x}{lg:02x}{lb:02x}"

    @staticmethod
    def _darken(hex_color: str, factor: float = 0.25) -> str:
        """Blend color towards black by 'factor' (0..1)."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        dr = int(r * (1 - factor))
        dg = int(g * (1 - factor))
        db = int(b * (1 - factor))
        return f"#{dr:02x}{dg:02x}{db:02x}"

    @staticmethod
    def _norm(s: str) -> str:
        return str(s).strip().casefold()

    @staticmethod
    def _bounds_tuple(bounds):
        # bounds is [minx, miny, maxx, maxy]
        return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    # centroid builders
    def _try_autobuild_centroids(self):
        """Build centroids from polygons if possible (via GeoPandas)."""
        if gpd is None:
            return
        try:
            # Prefer geodatasets (new standard) then legacy bundled datasets
            try:
                path = gpd.datasets.get_path("naturalearth_lowres")
            except Exception:
                if gds is None:
                    return
                path = gds.get_path("naturalearth.cultural.admin_0_countries")

            world = gpd.read_file(path)[["name", "continent", "geometry"]]
            w_proj = world.to_crs(3857)
            w_proj["rep"] = w_proj.geometry.representative_point()

            # countries centroids
            countries_rep = w_proj.set_geometry("rep").to_crs(4326)
            c = countries_rep.assign(
                country=countries_rep["name"],
                lat=countries_rep.geometry.y,
                lon=countries_rep.geometry.x,
            )[["country", "lat", "lon"]].copy()
            self._ensure_float_latlon(c)
            self._country_centroids = c

            # continents centroids
            cont = w_proj.dissolve(by="continent", as_index=True)
            cont["rep"] = cont.geometry.representative_point()
            continents_rep = cont.set_geometry("rep").to_crs(4326).reset_index()
            cc = continents_rep.assign(
                continent=continents_rep["continent"],
                lat=continents_rep.geometry.y,
                lon=continents_rep.geometry.x,
            )[["continent", "lat", "lon"]].copy()
            self._ensure_float_latlon(cc)
            self._continent_centroids = cc
        except Exception:
            pass  # leave fallbacks in place

    # continent centroids fallback
    def _continent_centroids_df(self) -> pd.DataFrame:
        if self._continent_centroids is not None:
            return self._continent_centroids.copy()
        # Minimal fallback (rough hand-picked)
        return pd.DataFrame(
            [
                ("Africa", 7.1881, 21.0938),
                ("Asia", 34.0479, 100.6197),
                ("Europe", 54.5260, 15.2551),
                ("North America", 54.5260, -105.2551),
                ("South America", -8.7832, -55.4915),
                ("Oceania", -22.7359, 140.0188),
            ],
            columns=["continent", "lat", "lon"],
        )
    # world polygons loader
    def _load_world_gdf(self):
        """
        Load/cached Natural Earth polygons:
        - Try GeoPandas legacy dataset, then geodatasets admin_0_countries.
        - As last resort, remote GeoJSON (no 'continent' column).
        """
        if self._world_gdf is not None:
            return self._world_gdf

        try:
            if gpd is None:
                raise RuntimeError("geopandas not available")

            # Try legacy bundled dataset
            try:
                path = gpd.datasets.get_path("naturalearth_lowres")
                gdf = gpd.read_file(path)
            except Exception:
                # Try geodatasets (preferred new way)
                if gds is not None:
                    path = gds.get_path("naturalearth.cultural.admin_0_countries")
                    gdf = gpd.read_file(path)
                else:
                    # Last fallback: remote (no continent)
                    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
                    gdf = gpd.read_file(url)

            keep = [c for c in ["name", "name_long", "name_en", "continent", "geometry"] if c in gdf.columns]
            gdf = gdf[keep].copy()

            # English display name
            if "name_en" in gdf.columns and gdf["name_en"].notna().any():
                gdf["display_name"] = gdf["name_en"]
            elif "name_long" in gdf.columns:
                gdf["display_name"] = gdf["name_long"]
            else:
                gdf["display_name"] = gdf["name"]

            gdf["name_norm"] = gdf["display_name"].astype(str).str.strip().str.casefold()

            self._world_gdf = gdf
            return gdf
        except Exception:
            self._world_gdf = None
            return None

    # map builder
    def build_map(
        self,
        df_counts: pd.DataFrame,
        level: str,                 # "country" or "continent"
        min_px: int = 5,
        max_px: int = 5,
        opacity: float = 0.75,
        use_sqrt: bool = True,
        cluster: bool = False,
        color_polygons: bool = True,
        selected_country: str | None = None,
        # base polygons (non-selected)
        polygon_base_color: str = "#9aa0a6",
        polygon_base_opacity: float = 0.15,
        polygon_border_color: str = "#6b7280",
        polygon_border_weight: int = 1,
        zoom_on_selected: bool = True,
        zoom_mode: str = "continent",   # "country" or "continent"
    ) -> folium.Map:
        """
        df_counts must have:
        - if level="country":   columns ['country','count_recipes']
        - if level="continent": columns ['continent','count_recipes']
        """

        # validate input
        req = ["country", "count_recipes"] if level == "country" else ["continent", "count_recipes"]
        missing_cols = [c for c in req if c not in df_counts.columns]
        if missing_cols:
            raise ValueError(f"df_counts must have {missing_cols} for level='{level}'")

        work = df_counts.copy()

        # attach centroids
        if level == "country":
            left = work.copy()
            left["country_norm"] = left["country"].astype(str).str.strip().str.casefold()
            right = self._country_centroids.copy()
            right["country_norm"] = right["country"].astype(str).str.strip().str.casefold()
            work = left.merge(right[["country_norm", "lat", "lon"]], on="country_norm", how="left")
            work["country"] = left["country"]
            work = work.drop(columns=["country_norm"])
            work = work.dropna(subset=["lat", "lon"])
            label_col = "country"
        else:
            cont = self._continent_centroids_df()
            work = work.merge(cont, on="continent", how="left")
            work = work.dropna(subset=["lat", "lon"])
            label_col = "continent"

        # sizes
        radii_px = self._scale_sizes(
            work["count_recipes"].to_numpy(),
            min_px=min_px, max_px=max_px, use_sqrt=use_sqrt
        )

        # selected row (for polygon + zoom)
        selected_row = None
        if selected_country and level == "country":
            sel = self._norm(selected_country)
            mask = work["country"].astype(str).str.strip().str.casefold() == sel
            if mask.any():
                selected_row = work[mask].iloc[0]

        # base map (ocean base + label overlay)
        center = [work["lat"].mean(), work["lon"].mean()] if len(work) else [20, 0]
        m = folium.Map(location=center, zoom_start=self.zoom_start, tiles=None, control_scale=self.control_scale)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
            attr="&copy; Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org, and others",
            name="Esri World Ocean Base",
            control=True,
        ).add_to(m)

        folium.map.CustomPane("labels", z_index=650).add_to(m)
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png",
            attr="&copy; OpenStreetMap contributors &copy; CARTO",
            name="CARTO Positron (labels)",
            overlay=True,
            control=True,
            opacity=0.95,
            pane="labels",
        ).add_to(m)

        layer = m if not cluster else MarkerCluster().add_to(m)

        # polygons (neutral base for all; bucket color ONLY for selected country)
        if color_polygons and level == "country":
            selected_norm = None
            selected_edge = polygon_border_color
            selected_fill = polygon_base_color
            selected_fill_opacity = 0.45

            if selected_row is not None and selected_country:
                v = int(selected_row["count_recipes"])
                selected_edge = self._seg_color(v)               # legend color
                selected_fill = self._lighten(selected_edge, 0.6)  # softer fill
                selected_norm = self._norm(selected_country)

            def _style(props: dict):
                # base look
                style = {
                    "color": polygon_border_color,
                    "weight": polygon_border_weight,
                    "fillColor": polygon_base_color,
                    "fillOpacity": polygon_base_opacity,
                }
                # selected override
                if selected_norm:
                    name = (
                        props.get("name") or props.get("NAME") or
                        props.get("name_long") or props.get("NAME_LONG") or
                        props.get("admin") or props.get("display_name") or ""
                    )
                    if self._norm(name) == selected_norm:
                        style.update({
                            "color": selected_edge,
                            "weight": 2,
                            "fillColor": selected_fill,
                            "fillOpacity": selected_fill_opacity,
                        })
                return style

            drew_polygons = False
            try:
                gdf = self._load_world_gdf()
                if gdf is not None:
                    folium.GeoJson(
                        data=gdf.__geo_interface__,
                        name="Countries (English)",
                        style_function=lambda feat: _style(feat.get("properties", {})),
                        tooltip=folium.GeoJsonTooltip(fields=["display_name"], aliases=["Country"], sticky=False),
                        control=False,
                    ).add_to(m)

                    # zoom logic
                    if selected_country and zoom_on_selected:
                        sel_norm = self._norm(selected_country)
                        match = gdf[gdf["name_norm"] == sel_norm]
                        if not match.empty:
                            if zoom_mode.lower() == "continent" and "continent" in gdf.columns:
                                cont_name = match.iloc[0].get("continent")
                                if pd.notna(cont_name):
                                    cont_geoms = gdf[gdf["continent"] == cont_name]
                                    if not cont_geoms.empty:
                                        bounds = cont_geoms.to_crs(4326).total_bounds
                                        m.fit_bounds(self._bounds_tuple(bounds))
                                    else:
                                        geom = match.iloc[0:1]
                                        bounds = geom.to_crs(4326).total_bounds
                                        m.fit_bounds(self._bounds_tuple(bounds))
                            else:
                                geom = match.iloc[0:1]
                                bounds = geom.to_crs(4326).total_bounds
                                m.fit_bounds(self._bounds_tuple(bounds))
                    drew_polygons = True
            except Exception:
                pass

            # Fallback: remote GeoJSON if GeoPandas not available (no continent zoom)
            if not drew_polygons:
                try:
                    WORLD_GEOJSON_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
                    folium.GeoJson(
                        data=WORLD_GEOJSON_URL,
                        name="Countries",
                        style_function=lambda feat: _style(feat.get("properties", {})),
                        control=False,
                    ).add_to(m)
                except Exception:
                    pass

        # bubbles with subtle glow + gloss
        for (_, row), r_px in zip(work.iterrows(), radii_px):
            value = int(row["count_recipes"])
            seg = self._seg_color(value)
            edge = self._darken(seg, 0.35)
            glow = self._lighten(seg, 0.65)

            label_title = f"{row[label_col]} — {value}"
            popup = folium.Popup(f"<b>{row[label_col]}</b><br/>Total : {value}", max_width=250)

            # 1) Outer "glow" ring
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=float(r_px * 1.8),
                color="transparent", weight=0,
                fill=True, fill_color=glow, fill_opacity=0.25,
                tooltip=label_title,
            ).add_to(layer)

            # 2) Subtle shadow ring
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=float(r_px * 1.25),
                color=edge, weight=2, opacity=0.25,
                fill=False,
            ).add_to(layer)

            # 3) Main colored circle
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=float(r_px),
                color=edge, weight=1, opacity=0.9,
                fill=True, fill_color=seg, fill_opacity=opacity,
                tooltip=label_title, popup=popup,
            ).add_to(layer)

            # 4) Glossy highlight (CSS radial gradient)
            highlight_size = max(10, r_px * 0.9)
            highlight_html = f"""
            <div style="
                position: relative;
                width: {highlight_size}px;
                height: {highlight_size}px;
                transform: translate(-50%, -65%);
                border-radius: 50%;
                pointer-events: none;
                background: radial-gradient(
                    circle at 35% 35%,
                    rgba(255,255,255,0.85) 0%,
                    rgba(255,255,255,0.35) 45%,
                    rgba(255,255,255,0.0) 70%
                );
            "></div>
            """
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(html=highlight_html),
            ).add_to(layer)

        # legend
        legend_html = """
        {% macro html(this, kwargs) %}
        <div style="
        position: fixed;
        top: 300px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 10px 12px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 12px;
        color: #000 !important;
        ">
          <div style="font-weight:700; margin-bottom:6px; color:#000 !important;">
            Recipes (bubble color)
          </div>
          <div style="color:#000 !important;">
            <span style="display:inline-block;width:12px;height:12px;background:#e03131;margin-right:6px;"></span>
            &lt; 4
          </div>
          <div style="color:#000 !important;">
            <span style="display:inline-block;width:12px;height:12px;background:#f59f00;margin-right:6px;"></span>
            5 – 14
          </div>
          <div style="color:#000 !important;">
            <span style="display:inline-block;width:12px;height:12px;background:#1c7ed6;margin-right:6px;"></span>
            15 – 29
          </div>
          <div style="color:#000 !important;">
            <span style="display:inline-block;width:12px;height:12px;background:#2f9e44;margin-right:6px;"></span>
            &gt; 30
          </div>
        </div>
        {% endmacro %}
        """
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

        folium.LayerControl(collapsed=True).add_to(m)

        return m