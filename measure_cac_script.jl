### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7879e86f-e900-45de-a7ba-5e768760d1b1
using DataFrames: DataFrame, rename!, nrow

# ╔═╡ c089c448-09c1-459c-8c88-07b021e1b6f5
using CSV: write

# ╔═╡ 7ede2cbc-c82a-40c4-86be-ac7c61c14f03
using StaticArrays: SVector

# ╔═╡ 8d24da97-a7ab-4fb8-b37b-c48c6c3d3f70
using IterTools: product

# ╔═╡ b8c49cef-c730-464b-b17c-005afdf0fdbe
using LinearAlgebra: normalize, dot

# ╔═╡ aa1f5386-f4f3-4bf6-a19d-41814fdd19f9
using CairoMakie: Figure, Axis, heatmap!, heatmap, scatter!, axislegend, hist!

# ╔═╡ 0cd17411-918f-4f02-b4a9-ffaa6de64bc8
using DICOM: dcmdir_parse

# ╔═╡ a7e5315b-adee-4168-a321-755a49016433
using PlutoUI: TableOfContents, Slider, TextField, confirm, combine

# ╔═╡ c4b10ab5-ef81-4617-be00-6c0e836dcdd8
using StatsBase: countmap, quantile

# ╔═╡ d39fe463-c2b7-4297-918a-490a31b4329e
using ImageMorphology: component_centroids, label_components, component_boxes, dilate

# ╔═╡ 5102b829-2ed0-430b-a942-72d96e21c197
using Statistics: mean, norm, std

# ╔═╡ bd259589-0cc3-40e2-a8e4-c3af710bf65a
using CalciumScoring: score, VolumeFraction, Agatston

# ╔═╡ 41d34cdb-af4c-4c59-9ae8-e020b72b6964
using ImageCore: channelview

# ╔═╡ 29fd1197-53c7-4660-849e-f1e684acd8d1
using DataInterpolations: QuadraticInterpolation

# ╔═╡ 10872202-4a25-4a8b-ab1d-35c4ece5a2a2
using OrderedCollections: OrderedDict

# ╔═╡ beda1edb-4fea-40d1-8bbc-c49953952698
md"""
# Set Up
"""

# ╔═╡ a214f633-3699-4fe6-bca4-02de420d3c1d
md"""
## Environment & Imports
"""

# ╔═╡ 4aff31dc-7fd7-4d89-9a31-26cd1133eeb5
TableOfContents()

# ╔═╡ f97c1ff0-36ae-4fcf-b7db-065c90ea13cf
md"""
## Helper Functions
"""

# ╔═╡ ae78dad9-4246-4c8b-87d3-d0ddc6f9316d
md"""
#### DICOM Utils
"""

# ╔═╡ a4aec909-8214-4615-8782-f4968c532742
function load_dcm_array(dcm_data)
    return array = cat(
        [dcm_data[i][(0x7fe0, 0x0010)] for i in eachindex(dcm_data)]...; dims=3
    )
end

# ╔═╡ b1ad4ce5-c28d-48bb-b249-3760a6acbaec
function get_pixel_size(header)
	head = copy(header)
	pixel_size = 
		try
			pixel_size = (head[(0x0028, 0x0030)])
            push!(pixel_size, head[(0x0018, 0x0050)])
		catch
			FOV = (head[(0x0018, 0x1100)])
			matrix_size = head[(0x0028, 0x0010)]
		
			pixel_size = FOV / matrix_size
            push!(pixel_size, head[(0x0018, 0x0050)])
		end
	return pixel_size
end

# ╔═╡ a95486b1-8cb3-4f03-b761-595bc24bd70c
md"""
#### Active Contours
"""

# ╔═╡ 5419e1ba-1709-4c4b-8a3d-edaef052cbcd
function initial_level_set(shape::Tuple{Int64, Int64})
    x₀ = reshape(collect(0:shape[begin]-1), shape[begin], 1)
    y₀ = reshape(collect(0:shape[begin+1]-1), 1, shape[begin+1])
    𝚽₀ = @. sin(pi / 5 * x₀) * sin(pi / 5 * y₀)
end

# ╔═╡ 42c07efe-f575-40a6-a65b-57b521061c5e
function initial_level_set(shape::Tuple{Int64, Int64, Int64})
    x₀ = reshape(collect(0:shape[begin]-1), shape[begin], 1, 1)
    y₀ = reshape(collect(0:shape[begin+1]-1), 1, shape[begin+1], 1)
    z₀ = reshape(collect(0:shape[begin+2]-1), 1, 1, shape[begin+2])
    𝚽₀ = @. sin(pi / 5 * x₀) * sin(pi / 5 * y₀) * sin(pi / 5 * z₀)
end

# ╔═╡ b7d72bb0-4ed8-4d78-bf47-4f333a2daee7
function calculate_averages(
	img::AbstractArray{T, N},
	H𝚽::AbstractArray{S, N},
	area::Int64, ∫u₀::Float64) where {T<:Real, S<:Bool, N}
    ∫u₀H𝚽 = 0
    ∫H𝚽 = 0
    @inbounds for i in eachindex(img)
        if H𝚽[i]
            ∫u₀H𝚽 += img[i]
            ∫H𝚽 += 1
        end
    end
    ∫H𝚽ⁱ = area - ∫H𝚽
    ∫u₀H𝚽ⁱ = ∫u₀ - ∫u₀H𝚽
    c₁ = ∫u₀H𝚽 / max(1, ∫H𝚽)
    c₂ = ∫u₀H𝚽ⁱ / max(1, ∫H𝚽ⁱ)

    return c₁, c₂
end

# ╔═╡ e5477a2e-9f40-41f3-888e-0c301302de64
function chan_vese(img;
                   μ::Real=0.25,
                   λ₁::Real=1.0,
                   λ₂::Real=1.0,
                   tol::Real=1e-3,
                   max_iter::Int=500,
                   Δt::Real=0.5,
                   normalize::Bool=false,
                   init_level_set=initial_level_set(size(img)))
	# Signs used in the codes and comments mainly follow paper[3] in the References.
	axes(img) == axes(init_level_set) || throw(ArgumentError("axes of input image and init_level_set should be equal. Instead they are $(axes(img)) and $(axes(init_level_set))."))
	img = Float64.(channelview(img))
	N = ndims(img)
	iter = 0
	h = 1.0
	del = tol + 1
	if normalize
		img .= img .- minimum(img)
		if maximum(img) != 0
			img .= img ./ maximum(img)
		end
	end
	
	# Precalculation of some constants which helps simplify some integration
	area = length(img)   # area = ∫H𝚽 + ∫H𝚽ⁱ
	∫u₀ = sum(img)       # ∫u₀ = ∫u₀H𝚽 + ∫u₀H𝚽ⁱ
	
	# Initialize the level set
	𝚽ⁿ = init_level_set
	
	# Preallocation and initializtion
	H𝚽 = trues(size(img)...)
	𝚽ⁿ⁺¹ = similar(𝚽ⁿ)
	
	Δ = ntuple(d -> CartesianIndex(ntuple(i -> i == d ? 1 : 0, N)), N)
	idx_first = first(CartesianIndices(𝚽ⁿ))
	idx_last = last(CartesianIndices(𝚽ⁿ))
	
	while (del > tol) & (iter < max_iter)
		ϵ = 1e-8
		diff = 0
	
		# Calculate the average intensities
		@. H𝚽 = 𝚽ⁿ > 0 # Heaviside function
		c₁, c₂ = calculate_averages(img, H𝚽, area, ∫u₀) # Compute c₁(𝚽ⁿ), c₂(𝚽ⁿ)
	
		# Calculate the variation of level set 𝚽ⁿ
		@inbounds @simd for idx in CartesianIndices(𝚽ⁿ)
			𝚽₀  = 𝚽ⁿ[idx] # 𝚽ⁿ(x, y)
			u₀ = img[idx]  # u₀(x, y)
			𝚽₊ = broadcast(i->𝚽ⁿ[i], ntuple(d -> idx[d] != idx_last[d]  ? idx + Δ[d] : idx, N))
			𝚽₋ = broadcast(i->𝚽ⁿ[i], ntuple(d -> idx[d] != idx_first[d] ? idx - Δ[d] : idx, N))
	
			# Solve the PDE of equation 9 in paper[3]
			center_diff = ntuple(d -> (𝚽₊[d] - 𝚽₋[d])^2 / 4., N)
			sum_center_diff = sum(center_diff)
			C₊ = ntuple(d -> 1. / sqrt(ϵ + (𝚽₊[d] - 𝚽₀)^2 + sum_center_diff - center_diff[d]), N)
			C₋ = ntuple(d -> 1. / sqrt(ϵ + (𝚽₋[d] - 𝚽₀)^2 + sum_center_diff - center_diff[d]), N)
	
			K = sum(𝚽₊ .* C₊) + sum(𝚽₋ .* C₋)
			δₕ = h / (h^2 + 𝚽₀^2) # Regularised Dirac function
			difference_from_average = - λ₁ * (u₀ - c₁) ^ 2 + λ₂ * (u₀ - c₂) ^ 2
	
			𝚽ⁿ⁺¹[idx] = 𝚽 = (𝚽₀ + Δt * δₕ * (μ * K + difference_from_average)) / (1. + μ * Δt * δₕ * (sum(C₊) + sum(C₋)))
			diff += (𝚽 - 𝚽₀)^2
		end
	
		del = sqrt(diff / area)
	
		# Reinitializing the level set is not strictly necessary, so this part of code is commented.
		# If you wants to use the reinitialization, just uncommented codes following.
		# Function `reinitialize!` is already prepared.
	
		# reinitialize!(𝚽ⁿ⁺¹, 𝚽ⁿ, Δt, h) # Reinitialize 𝚽 to be the signed distance function to its zero level set
	
		𝚽ⁿ .= 𝚽ⁿ⁺¹
	
		iter += 1
	end
	
	return 𝚽ⁿ .> 0
end

# ╔═╡ 26ff0532-dbf6-485d-a83f-199d0f70b0e6
md"""
#### Masks
"""

# ╔═╡ 0a61db85-1f29-4b90-a4da-651f15b0f38b
function create_mask(array, mask)
    @assert size(array) == size(mask)
    idxs = findall(x -> x == true, mask)
    overlayed_mask = zeros(size(array))
    for idx in idxs
        overlayed_mask[idx] = array[idx]
    end
    return overlayed_mask
end

# ╔═╡ aae0aaec-8268-4765-a451-0bceb31dd6e2
function calculate_thresholds(kV, mA)
	if kV == 80
		mA_arr = [10, 20, 40, 100, 150, 250]
		threshold_low_arr = [100, 80, 50, 20, 10, 0]
		threshold_high_arr = [300, 255, 180, 90, 65, 40]
    elseif kV == 100
		mA_arr = [10, 20, 40, 100, 150, 250]
		threshold_low_arr = [70, 55, 35, 35, 29, 25]
		threshold_high_arr = [230, 150, 90, 88, 83, 80]
    elseif kV == 120
		mA_arr = [10, 15, 20, 25, 30, 40, 50, 100, 150, 250]
		threshold_low_arr = [40, 40, 39, 39, 39, 39, 39, 35, 23, 22]
		threshold_high_arr = [150, 110, 105, 100, 100, 95, 90, 90, 88, 85]
    end

	threshold_low_interp = QuadraticInterpolation(threshold_low_arr, mA_arr; extrapolate = true)
	threshold_high_interp = QuadraticInterpolation(threshold_high_arr, mA_arr; extrapolate = true)

	threshold_low = threshold_low_interp(mA)
	threshold_high = threshold_high_interp(mA)
    return threshold_low, threshold_high
end

# ╔═╡ 172d043b-c600-43e6-a75d-524bc9dd4955
function threshold_low_high(dcm_arr, kV, mA)
    threshold_low, threshold_high = calculate_thresholds(kV, mA)
	thresholded_mask_low = dcm_arr .> threshold_low
    thresholded_mask_high = dcm_arr .< threshold_high
    masked_thresholded = thresholded_mask_high .& thresholded_mask_low

    return masked_thresholded
end

# ╔═╡ 18e0ba73-e0f5-42b5-a915-d9287bd9a006
md"""
#### Orthanc Tools
"""

# ╔═╡ bf3d544f-9ab8-430a-b7cb-5da5c7381d52
import HTTP

# ╔═╡ 2861ed2a-adfe-4375-a2cb-0d090f8f1e02
import JSON

# ╔═╡ 89354634-0bec-4d40-a71a-d48f3d9727a1
function get_all_studies(ip_address::String="localhost"; show_warnings=false)
	url_studies = HTTP.URI(scheme="http", host=ip_address, port="8042", path="/studies")
    studies = JSON.parse(String(HTTP.get(url_studies).body))

    studies_dict = OrderedDict{String,Vector}()
    for study in studies
        s = JSON.parse(String(HTTP.get(string(url_studies, "/", study)).body))
        try
            accession_num = s["MainDicomTags"]["AccessionNumber"]
            if !haskey(studies_dict, accession_num)
                push!(studies_dict, accession_num => [study])
            else
                push!(studies_dict[accession_num], study)
            end
        catch
			if show_warnings
            	@warn "No accession number for $study"
			end
        end
    end

    return studies_dict
end

# ╔═╡ b139733c-638e-40e5-877e-5b2033838f18
function get_all_series(
    studies_dict::OrderedDict,
    accession_num::String,
    ip_address::String="localhost")

	url_study = HTTP.URI(scheme="http", host=ip_address, port="8042", path="/studies/$(studies_dict[accession_num]...)")

    series = JSON.parse(String(HTTP.get(url_study).body))
    series_dict = OrderedDict{String,Vector}()
    for ser in series["Series"]
		url_series = HTTP.URI(
			scheme="http", host=ip_address, port="8042", path=string("/series/", ser)
		)
        s = JSON.parse(String(HTTP.get(url_series).body))
        try
            series_num = s["MainDicomTags"]["SeriesNumber"]
            if !haskey(series_dict, series_num)
                push!(series_dict, series_num => [ser])
            else
                push!(series_dict[series_num], ser)
            end
        catch
            @warn "No series number for $ser"
        end
    end

    return series_dict
end

# ╔═╡ a75ef71d-c5de-4614-bb76-09355919d783
function get_all_instances(
    series_dict::OrderedDict,
    series_num::String,
    ip_address::String="localhost")

    url = HTTP.URI(scheme="http", host=ip_address, port="8042")
    instances_dict = OrderedDict{String,Vector}()
    for ser in series_dict[series_num]
		url_series = HTTP.URI(
			scheme="http", host=ip_address, port="8042", path=string("/series/", ser)
		)
        series = JSON.parse(String(HTTP.get(url_series).body))
        instances = series["Instances"]
        if !haskey(instances_dict, series_num)
            push!(instances_dict, series_num => [instances])
        else
            push!(instances_dict[series_num], instances)
        end
    end
    return instances_dict
end

# ╔═╡ 08774908-cb99-4673-8470-7881301da059
function download_instances(
	instances_dict::OrderedDict,
	instance_num::Number,
	output_directory::String,
	ip_address::String="localhost")

	for (key, value) in instances_dict
		for i in value[instance_num]
			url_instance = string("http://", ip_address, ":8042", string("/instances/", i))
			instance = JSON.parse(String(HTTP.get(url_instance).body))
			idx = instance["IndexInSeries"]
			download(string(url_instance, "/", "file"), joinpath(output_directory, "$(idx).dcm"))
		end
	end
	@info "Files located at $(output_directory)"
end

# ╔═╡ 69c5661c-77c0-4b6b-b0bc-3cf1512b7c60
function process_instances(
	instances_dicts, series_num_vec, output_dir, instance_number, ip_address
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    output_paths = String[]

	for (idx, dict) in enumerate(instances_dicts)
		_series = collect(keys(dict))[1]
		
	    path = joinpath(output_dir, string(series_num_vec[idx]))
	    if !isdir(path)
	        mkpath(path)
	    end
		
	    download_instances(instances_dicts[idx], instance_number, path, ip_address)
	
	    output_path = joinpath(output_dir, string(_series))
		push!(output_paths, output_path)
	end

    return output_paths
end

# ╔═╡ da41d9b3-e124-4ef8-81e0-d3e5eea52337
md"""
#### Mask Heart Function
"""

# ╔═╡ 17f9ba72-aa80-4ff5-9de2-83512b6c49ce
function erode_mask(img, num_erosions)
	new_img = img
	i = 0
	while i < num_erosions
		new_img = erode(new_img)
		i += 1
	end
	return new_img
end

# ╔═╡ 7b84a208-0a59-4aeb-9506-df5eff303305
function create_circle_mask(img, centroids, radius)
    # initialize mask with all zeros
    mask = zeros(size(img))

    # define the center of the circle
    x0, y0 = centroids[1], centroids[2]

    # set all pixels inside the circle to 1 and all pixels outside the circle to 0
    for x in axes(img, 1), y in axes(img, 2)
        if ((x - x0)^2 + (y - y0)^2) <= radius^2
            mask[x, y] = 1
        else
            mask[x, y] = 0
        end
    end
    return Bool.(mask)
end

# ╔═╡ cdb712cf-12ee-4e9d-9d04-16f4af14bbd8
function centroids_from_mask(mask)
	cc_labels = label_components(mask)
	largest_connected_component, _ = sort(collect(pairs(countmap(cc_labels[cc_labels .!= 0]))), by=x->x[2], rev=true)
	largest_connected_indices = findall(cc_labels .== largest_connected_component[1])

	new_mask = zeros(size(mask))
	for i in largest_connected_indices
		new_mask[i] = 1
	end
	centroids = Int.(round.(component_centroids(label_components(new_mask))[end]))
end

# ╔═╡ 3f695cba-2572-4a50-b76a-80fec1056692
function create_initial_level_set(dcm_arr)
	half_x, half_y = size(dcm_arr, 1) ÷ 2, size(dcm_arr, 2) ÷ 2
	init_circle = create_circle_mask(dcm_arr[:, :, 3], (half_x, half_y), 140)
	
	init_mask = BitArray(undef, size(dcm_arr))
	for z in axes(dcm_arr, 3)
		init_mask[:, :, z] = init_circle
	end
	init_mask = init_mask .* initial_level_set(size(init_mask))

	return init_mask
end

# ╔═╡ c33aaaa0-6a87-41b4-86ed-669016382b6e
md"""
#### Inserts
"""

# ╔═╡ c9648ed4-44a2-41c0-b019-e8d68e2a7a6a
function get_insert_centers(dcm, threshold)
	dcm_slice_thresh = dcm .> threshold
	
	# Use connected component labeling to identify and label all connected components
	cc_labels = label_components(dcm_slice_thresh)

	# Use the countmap function to count the number of occurrences of each value in the array, excluding 0
	counts = countmap(cc_labels[cc_labels .!= 0])
	
	# Find the value with the most occurrences
	most_common_value_a, most_common_value_b = sort(collect(pairs(counts)), by=x->x[2], rev=true)

	# Find the indices of the most common value in the original array
	most_common_indices_a = findall(cc_labels .== most_common_value_a[1])

	# Create boolean array from new cartesian indices
	bool_arr_a = zeros(size(dcm_slice_thresh))
	for i in most_common_indices_a
		bool_arr_a[i] = 1
	end
	centroids_a = Int.(round.(component_centroids(label_components(bool_arr_a))[end]))
	box_a = component_boxes(label_components(bool_arr_a))

	# Find the indices of the most common value in the original array
	most_common_indices_b = findall(cc_labels .== most_common_value_b[1])

	# Create boolean array from new cartesian indices
	bool_arr_b = zeros(size(dcm_slice_thresh))
	for i in most_common_indices_b
		bool_arr_b[i] = 1
	end
	centroids_b = Int.(round.(component_centroids(label_components(bool_arr_b))[end]))

	# centers_a, centers_b = [centroids_a..., z], [centroids_b..., z]
	centers_a, centers_b = centroids_a, centroids_b
	return centers_a, centers_b
	
end

# ╔═╡ e62f58f7-141f-4c1a-94e0-561d95eec28b
# Modify the in_cylinder function to accept Static Vectors
function _in_cylinder(pt::SVector{3, Int}, pt1::SVector{3, Float64}, pt2::SVector{3, Float64}, radius)
    v = pt2 - pt1
    w = pt - pt1

    # Compute the dot product
    c1 = dot(w, v)
    if c1 <= 0
        return norm(w) <= radius
    end

    c2 = dot(v, v)
    if c2 <= c1
        return norm(pt - pt2) <= radius
    end

    # Compute the perpendicular distance
    b = c1 / c2
    pb = pt1 + b * v
    return norm(pt - pb) <= radius
end

# ╔═╡ 0b72af17-61fc-4b86-a338-8f86d32058fe
function create_cylinder(array, pt1, pt2, radius, offset)
    # Convert the points to static arrays
    pt1 = SVector{3, Float64}(pt1)
    pt2 = SVector{3, Float64}(pt2)

    # Compute the unit vector in the direction from pt1 to pt2
    direction = normalize(pt2 - pt1)

    # Adjust the endpoints of the cylinder by the offset
    pt1 = pt1 - offset * direction
    pt2 = pt2 + offset * direction

    # Initialize the 3D array
    cylinder = zeros(Int, size(array)...)
    # Iterate over the 3D array
    for k in axes(cylinder, 3)
        for j in axes(cylinder, 2)
            for i in axes(cylinder, 1)
                # Create a static vector for the current point
                pt = SVector{3, Int}(i, j, k)

                # Check if the current point is inside the cylinder
                if _in_cylinder(pt, pt1, pt2, radius)
                    cylinder[i, j, k] = 1
                end
            end
        end
    end
    return Bool.(cylinder)
end

# ╔═╡ d61f577c-531a-4f16-a806-5bcfc64a94c2
function remove_outliers(vector)
    Q1 = quantile(vector, 0.25)
    Q3 = quantile(vector, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    return [x for x in vector if x > lower_bound]
end

# ╔═╡ da4739f0-4857-4b12-a5a9-bab91c543694
md"""
## Docs
"""

# ╔═╡ d6f176df-b323-4d74-b609-09a50b82a15d
md"""
!!! info ""
	See `README.md` in the root directory for more info on the scans

	| Accession Number | Scan Name | Series (80 kV) | Series (100 kV) | Series (120 kV) |
	| ---------------- | --------- | -------------- | --------------- | --------------- |
	| 3082             | C_0bpm    | 2-11           | 12-21           | 22-31           |
	| 3083             | F_0bpm    | 2-11           | 12-21           | 22-31           |
	| 3087             | E_0bpm    | 2-11           | 12-21           | 22-31           |
	| 3095             | D_0bpm    | 2-11           | 12-21           | 22-31           |
"""

# ╔═╡ 4f0f5767-5c26-4ae4-a28c-20ca9ed986ee
md"""
# Download From Orthanc
"""

# ╔═╡ 96c15de9-3100-4d4d-b2e0-1380cf0ec31f
md"""
## Get Studies
Insert the IP address associated with the Orthanc server into the input box below and then click "Submit". When the code is finished, you can inspect the files by clicking on the dictionary.
"""

# ╔═╡ 1c611a7b-a89c-4bd3-99cc-296a4ccf43a5
@bind ip_address confirm(TextField(default="128.200.49.26"))

# ╔═╡ af303018-55b9-4330-8e8d-a3e7d1d98697
studies_dict = get_all_studies(ip_address)

# ╔═╡ 6dae0032-6a60-4421-a68e-a94c44416e6e
md"""
## Get Series
Insert the accession number into the input box above and click "Submit". When the code is finished, you can inspect the files by clicking on the dictionary.
"""

# ╔═╡ 6030b8c8-d10b-442b-875c-c9e8f5bd016a
md"""
## Get Instance(s)
You can insert the series number of interest into the input box above and then click "Submit". When the code is finished, you can inspect the files by clicking on the dictionary.
"""

# ╔═╡ b7807c48-6abe-416e-bbd9-5922e9044f33
md"""
# Download DICOM Instance(s)
Type the folder path above, where you want the DICOM files to be saved (or use a temporary directory via `mktempdir()`) in the code cell below. Then type in the instance number that you want to download and click "Submit".
"""

# ╔═╡ 4beb3247-28f0-4b30-b925-0061253ee304
output_dir_temp = mktempdir() # r

# ╔═╡ 87bc30a2-7b7e-4877-8618-aac5b44f88db
function download_info(acc, ser, inst, save_folder_path)
	
	return combine() do Child
		
		inputs = [
			md""" $(acc): $(
				Child(TextField(default="3082"))
			)""",
			md""" $(ser): $(
				Child(TextField(default="2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31"))
			)""",
			md""" $(inst): $(
				Child(TextField(default="1"))
			)""",
			md""" $(save_folder_path): $(
				Child(TextField(default=output_dir_temp)))
			)"""
		]
		
		md"""
		#### Scan Details
		Input the relevant DICOM information to download the appropriate scans
		$(inputs)
		"""
	end
end

# ╔═╡ bb89a69b-35fc-4b42-a36c-9ac7640476c8
@bind details confirm(download_info("Accession Number", "Series Number(s)", "Instance Number", "Output Directory"))

# ╔═╡ e0e10a0c-464b-4185-bff7-d0a4e577cae1
accession_number, series_num, instance_num, output_dir = details

# ╔═╡ a23a7779-6b79-444c-a172-33afdf5fa204
series_dict = get_all_series(studies_dict, accession_number, ip_address)

# ╔═╡ 850df889-4876-4018-bf79-9de790d10091
series_num_vec = parse.(Int, split(series_num, ","))

# ╔═╡ 061edd0d-31a1-4f9a-842e-9b3c113d6e4b
begin
	instances_dicts = []
	for i in series_num_vec
		instances_dict = get_all_instances(series_dict, string(i), ip_address)
		push!(instances_dicts, instances_dict)
	end
end

# ╔═╡ 1cbc8875-8b23-4660-8eb6-543f31436c73
instances_dicts

# ╔═╡ 986a7b15-a441-412f-88a7-25f9648cfcbc
instance_number = parse(Int64, instance_num)

# ╔═╡ d60685cd-846f-4591-9908-2f2275a193d0
instances_dicts

# ╔═╡ 08e918cd-0032-4f8e-be16-84f2b19d8291
md"""
# Run Script
"""

# ╔═╡ b22040f6-5726-4551-af1d-b65ae876e83c
begin
	results_df = DataFrame(
		insert_name = String[],
		beats_per_minute = String[],
		kV = Float64[],
		mA = Float64[],
		ctdi = Float64[],
		gt_mass = Float64[],
		vf_mass = Float64[],
		vf_mass_bkg_mean = Float64[],
		vf_mass_bkg_std = Float64[],
		agatston_mass = Float64[],
		agatston_mass_bkg_mean = Float64[],
		agatston_mass_bkg_std = Float64[]
	)

	local insert_name
	local beats_per_minute
	for (i, dicts) in enumerate(instances_dicts)
		@info i
		output_paths = process_instances([dicts], series_num_vec[i], output_dir, instance_number, ip_address)
		
		output_path = output_paths[end]
	
		# Load DICOMs
		dcms = dcmdir_parse(output_path)
		dcm_arr = load_dcm_array(dcms)
	
		header = dcms[1].meta
		scan_name = header[(0x0010, 0x0020)]
		insert_name = split(scan_name, "_")[1]
		beats_per_minute = split(scan_name, "_")[2]
		
		mA = header[(0x0018, 0x1151)]
		kV = header[(0x0018, 0x0060)]
		ctdi = header[(0x0018, 0x9345)] # mGy
		x_space, y_space = header[(0x0028, 0x0030)]
		slice_thickness = header[(0x0018, 0x0050)]
		pixel_size = [x_space, y_space, slice_thickness]
	
		# # Mask Heart
		# mask_thresholded = threshold_low_high(dcm_arr, kV, mA)
		# centroids = centroids_from_mask(mask_thresholded)

		centroids = div.(size(dcm_arr), 2)
		heart_rad = 95
		heart_mask = create_circle_mask(dcm_arr[:, :, 3], centroids, heart_rad)
	
		## Extract density
		if insert_name == "A" || insert_name == "B" || insert_name == "C"
			gt_density = 0.050  # mg/mm^3
		elseif insert_name == "D" || insert_name == "E" || insert_name == "F"
			gt_density = 0.100  # mg/mm^3
		end

		## Extract diameter
		if insert_name == "A" || insert_name == "D"
			diameter = 1.2 # mm
		elseif insert_name == "B" || insert_name == "E"
			diameter = 3.0 # mm
		elseif insert_name == "C" || insert_name == "F"
			diameter = 5.0 # mm
		end

		# Inserts
		dcm_heart = dcm_arr .* heart_mask
		insert_threshold = 500
		centers_a, centers_b = get_insert_centers(dcm_heart, insert_threshold)
		cylinder_rad = diameter * 2.0
		cylinder = create_cylinder(dcm_heart, centers_a, centers_b, cylinder_rad, -25)
		background_rad = cylinder_rad + 6
		_background_ring = create_cylinder(dcm_heart, centers_a, centers_b, background_rad, -25)
		background_ring = Bool.(_background_ring .- cylinder)
	
		binary_calibration = falses(size(dcm_heart))
		binary_calibration[centers_a...] = true
		binary_calibration = dilate(binary_calibration)
		cylinder_clean = remove_outliers(dcm_heart[cylinder])
		background_clean = remove_outliers(dcm_heart[background_ring])
		
		# Offset Measurements
		off_num = 30
		offa1 = (centers_a[1] + off_num, centers_a[2] + off_num, centers_a[3])
		offb1 = (centers_b[1] + off_num, centers_b[2] + off_num, centers_b[3])
		offa2 = (centers_a[1] - off_num, centers_a[2] - off_num, centers_a[3])
		offb2 = (centers_b[1] - off_num, centers_b[2] - off_num, centers_b[3])
		offa3 = (centers_a[1] + off_num, centers_a[2] - off_num, centers_a[3])
		offb3 = (centers_b[1] + off_num, centers_b[2] - off_num, centers_b[3])
		
		offset_cylinder1 = create_cylinder(dcm_heart, offa1, offb1, cylinder_rad, -25)
		offset_cylinder2 = create_cylinder(dcm_heart, offa2, offb2, cylinder_rad, -25)
		offset_cylinder3 = create_cylinder(dcm_heart, offa3, offb3, cylinder_rad, -25)
		
		offset_background_ring1 = Bool.(create_cylinder(dcm_heart, offa1, offb1, background_rad, -25) .- offset_cylinder1)
		offset_background_ring2 = Bool.(create_cylinder(dcm_heart, offa2, offb2, background_rad, -25) .- offset_cylinder2)
		offset_background_ring3 = Bool.(create_cylinder(dcm_heart, offa3, offb3, background_rad, -25) .- offset_cylinder3)
		
		offset_cylinder1_clean = remove_outliers(dcm_heart[offset_cylinder1])
		offset_cylinder2_clean = remove_outliers(dcm_heart[offset_cylinder2])
		offset_cylinder3_clean = remove_outliers(dcm_heart[offset_cylinder3])
		
		offset_background_ring1_clean = remove_outliers(dcm_heart[offset_background_ring1])
		offset_background_ring2_clean = remove_outliers(dcm_heart[offset_background_ring2])
		offset_background_ring3_clean = remove_outliers(dcm_heart[offset_background_ring3])
		
		# Ground truth mass
		num_inserts = 3
		gt_length = 7 # mm
		gt_volume = π * (diameter/2)^2 * gt_length * num_inserts # mm^3
		gt_mass = gt_density * gt_volume
	
		# Volume fraction mass
		## Extract "calibration" from the endpoint inserts (ρ: 400 mg/cc)
		hu_calcium_400 = mean(dcm_heart[binary_calibration])
		ρ_calcium_400 = 0.400 # mg/mm^3

		## Calculate
		voxel_size = pixel_size[1] * pixel_size[2] * pixel_size[3]
		hu_heart_tissue_bkg = mean(background_clean)
		hu_heart_tissue_bkg_offset1 = mean(offset_background_ring1_clean)
		hu_heart_tissue_bkg_offset2 = mean(offset_background_ring2_clean)
		hu_heart_tissue_bkg_offset3 = mean(offset_background_ring3_clean)
		
		vf_mass = score(cylinder_clean, hu_calcium_400, hu_heart_tissue_bkg, voxel_size, ρ_calcium_400, VolumeFraction())
		vf_mass_offset1 = score(offset_cylinder1_clean, hu_calcium_400, hu_heart_tissue_bkg_offset1, voxel_size, ρ_calcium_400, VolumeFraction())
		vf_mass_offset2 = score(offset_cylinder2_clean, hu_calcium_400, hu_heart_tissue_bkg_offset2, voxel_size, ρ_calcium_400, VolumeFraction())
		vf_mass_offset3 = score(offset_cylinder3_clean, hu_calcium_400, hu_heart_tissue_bkg_offset3, voxel_size, ρ_calcium_400, VolumeFraction())
		
		vf_mass_bkg_mean = mean([vf_mass_offset1, vf_mass_offset2, vf_mass_offset3])
		vf_mass_bkg_std = std([vf_mass_offset1, vf_mass_offset2, vf_mass_offset3])
	
		# Agatston
		mass_cal_factor = ρ_calcium_400 / hu_calcium_400
		agatston_agatston, agatston_volume, agatston_mass = score(cylinder_clean, pixel_size, mass_cal_factor, Agatston(); kV=kV)
		
		_, _, agatston_mass_offset1 = score(offset_cylinder1_clean, pixel_size, mass_cal_factor, Agatston(); kV=kV)
		_, _, agatston_mass_offset2 = score(offset_cylinder2_clean, pixel_size, mass_cal_factor, Agatston(); kV=kV)
		_, _, agatston_mass_offset3 = score(offset_cylinder3_clean, pixel_size, mass_cal_factor, Agatston(); kV=kV)
		
		agatston_mass_bkg_mean = mean([agatston_mass_offset1, agatston_mass_offset2, agatston_mass_offset3])
		agatston_mass_bkg_std = std([agatston_mass_offset1, agatston_mass_offset2, agatston_mass_offset3])

		@info """
		Ground Truth Mass: $(gt_mass) \n
		Volume Fraction Mass: $(vf_mass) \n
		Volume Fraction Background Mass Mean + Std: $((vf_mass_bkg_mean, vf_mass_bkg_std)) \n
		Agatston Mass: $(agatston_mass) \n
		Agatston Background Mass Mean + Std: $((agatston_mass_bkg_mean, agatston_mass_bkg_std))
		"""
	
		# Push results to DataFrame
		push!(results_df, [insert_name, beats_per_minute, kV, mA, ctdi, gt_mass, vf_mass, vf_mass_bkg_mean, vf_mass_bkg_std, agatston_mass, agatston_mass_bkg_mean, agatston_mass_bkg_std])
	end

	output_filename = joinpath(pwd(),"data","$(insert_name)_$(beats_per_minute).csv")
	write(output_filename, results_df)
end

# ╔═╡ 4951072e-5546-4825-a12c-bb917df5995c
results_df

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CalciumScoring = "9c0cb1da-21b1-4615-967b-153e03110a28"
DICOM = "a26e6606-dd52-5f6a-a97f-4f611373d757"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataInterpolations = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
ImageCore = "a09fc81d-aa75-5fe9-8630-4744c3626534"
ImageMorphology = "787d08f9-d448-5407-9aad-5290dd7ab264"
IterTools = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.14"
CairoMakie = "~0.12.2"
CalciumScoring = "~0.4.0"
DICOM = "~0.11.0"
DataFrames = "~1.6.1"
DataInterpolations = "~5.2.0"
HTTP = "~1.10.8"
ImageCore = "~0.10.2"
ImageMorphology = "~0.4.5"
IterTools = "~1.10.0"
JSON = "~0.21.4"
OrderedCollections = "~1.6.3"
PlutoUI = "~0.7.59"
StaticArrays = "~1.9.4"
StatsBase = "~0.34.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "efb9f0687ed4a560919717b6c1ac6293153a8010"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ed2ec3c9b483842ae59cd273834e5b46206d6dda"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.11.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "585a387a490f1c4bd88be67eea15b93da5e85db7"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.5"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "9e8eaaff3e5951d8c61b7c9261d935eb27e0304b"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.CalciumScoring]]
deps = ["DSP", "Distributions", "ImageMorphology", "Statistics", "Unitful"]
git-tree-sha1 = "afa42971a3767e379b6f27333973ef18647d61d1"
uuid = "9c0cb1da-21b1-4615-967b-153e03110a28"
version = "0.4.0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DICOM]]
git-tree-sha1 = "8857acfd0f2ae6f2d27a187fb7cdcd990e0866da"
uuid = "a26e6606-dd52-5f6a-a97f-4f611373d757"
version = "0.11.0"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "f7f4319567fe769debfcf7f8c03d8da1dd4e2fb0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.9"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataInterpolations]]
deps = ["FindFirstFunctions", "ForwardDiff", "LinearAlgebra", "PrettyTables", "RecipesBase", "Reexport"]
git-tree-sha1 = "a47492f3694b8cd647a9a172a5111f585868f2c6"
uuid = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
version = "5.2.0"

    [deps.DataInterpolations.extensions]
    DataInterpolationsChainRulesCoreExt = "ChainRulesCore"
    DataInterpolationsOptimExt = "Optim"
    DataInterpolationsRegularizationToolsExt = "RegularizationTools"
    DataInterpolationsSymbolicsExt = "Symbolics"

    [deps.DataInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    RegularizationTools = "29dad682-9a27-4bc3-9c72-016788665182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "1755070db557ec2c37df2664c75600298b0c1cfc"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.0.3"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
git-tree-sha1 = "e90fef90f7d75e6a5b435b0fd65609759f99717a"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "1.2.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "801aef8228f7f04972e596b09d4dba481807c913"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "fc713f007cff99ff9e50accba6373624ddd33588"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "90709228dc114e599a2b62b7d23482a4f50938ee"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.13"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8f6786d8b2b3248d79db3ad359ce95382d5a6df8"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.170"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "ec3a60c9de787bc6ef119d13e07d4bfacceebb83"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.2"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c1c9da1a69f6c635a60581c98da252958c844d70"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.2"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "1865d0b8a2d91477c8b16b49152a32764c7b1f5f"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cb5a2ab6763464ae0f19c86c56c63d4a2b0f5bda"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "25e7f73d679e5214971620886d3416c1f5991ecc"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.9"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "2803cab51702db743f3fda07dd1745aadfbf43bd"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.5.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "90b4f68892337554d31cdcdbe19e48989f26c7e6"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.3"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "9ae599cd7529cfce7fea36cf00a62cfc56f0f37c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.4"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "bc7fd5c91041f44636b2c134041f7e5263ce58ae"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.0"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "a947ea21087caba0a798c5e494d0bb78e3a1a3a0"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.9"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "dd260903fdabea27d9b6021689b3cd5401a57748"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.20.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "e863582a41c5731f51fd050563ae91eb33cf09be"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.68"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "52ff2af32e591541550bd753c0da8b9bc92bb9d9"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.7+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─beda1edb-4fea-40d1-8bbc-c49953952698
# ╟─a214f633-3699-4fe6-bca4-02de420d3c1d
# ╠═7879e86f-e900-45de-a7ba-5e768760d1b1
# ╠═c089c448-09c1-459c-8c88-07b021e1b6f5
# ╠═7ede2cbc-c82a-40c4-86be-ac7c61c14f03
# ╠═8d24da97-a7ab-4fb8-b37b-c48c6c3d3f70
# ╠═b8c49cef-c730-464b-b17c-005afdf0fdbe
# ╠═aa1f5386-f4f3-4bf6-a19d-41814fdd19f9
# ╠═0cd17411-918f-4f02-b4a9-ffaa6de64bc8
# ╠═a7e5315b-adee-4168-a321-755a49016433
# ╠═c4b10ab5-ef81-4617-be00-6c0e836dcdd8
# ╠═d39fe463-c2b7-4297-918a-490a31b4329e
# ╠═5102b829-2ed0-430b-a942-72d96e21c197
# ╠═bd259589-0cc3-40e2-a8e4-c3af710bf65a
# ╠═41d34cdb-af4c-4c59-9ae8-e020b72b6964
# ╠═4aff31dc-7fd7-4d89-9a31-26cd1133eeb5
# ╟─f97c1ff0-36ae-4fcf-b7db-065c90ea13cf
# ╟─ae78dad9-4246-4c8b-87d3-d0ddc6f9316d
# ╠═a4aec909-8214-4615-8782-f4968c532742
# ╠═b1ad4ce5-c28d-48bb-b249-3760a6acbaec
# ╟─a95486b1-8cb3-4f03-b761-595bc24bd70c
# ╠═e5477a2e-9f40-41f3-888e-0c301302de64
# ╠═5419e1ba-1709-4c4b-8a3d-edaef052cbcd
# ╠═42c07efe-f575-40a6-a65b-57b521061c5e
# ╠═b7d72bb0-4ed8-4d78-bf47-4f333a2daee7
# ╟─26ff0532-dbf6-485d-a83f-199d0f70b0e6
# ╠═0a61db85-1f29-4b90-a4da-651f15b0f38b
# ╠═29fd1197-53c7-4660-849e-f1e684acd8d1
# ╠═aae0aaec-8268-4765-a451-0bceb31dd6e2
# ╠═172d043b-c600-43e6-a75d-524bc9dd4955
# ╟─18e0ba73-e0f5-42b5-a915-d9287bd9a006
# ╠═10872202-4a25-4a8b-ab1d-35c4ece5a2a2
# ╠═bf3d544f-9ab8-430a-b7cb-5da5c7381d52
# ╠═2861ed2a-adfe-4375-a2cb-0d090f8f1e02
# ╠═89354634-0bec-4d40-a71a-d48f3d9727a1
# ╠═b139733c-638e-40e5-877e-5b2033838f18
# ╠═a75ef71d-c5de-4614-bb76-09355919d783
# ╠═08774908-cb99-4673-8470-7881301da059
# ╠═69c5661c-77c0-4b6b-b0bc-3cf1512b7c60
# ╟─da41d9b3-e124-4ef8-81e0-d3e5eea52337
# ╠═17f9ba72-aa80-4ff5-9de2-83512b6c49ce
# ╠═7b84a208-0a59-4aeb-9506-df5eff303305
# ╠═cdb712cf-12ee-4e9d-9d04-16f4af14bbd8
# ╠═3f695cba-2572-4a50-b76a-80fec1056692
# ╟─c33aaaa0-6a87-41b4-86ed-669016382b6e
# ╠═c9648ed4-44a2-41c0-b019-e8d68e2a7a6a
# ╠═e62f58f7-141f-4c1a-94e0-561d95eec28b
# ╠═0b72af17-61fc-4b86-a338-8f86d32058fe
# ╠═d61f577c-531a-4f16-a806-5bcfc64a94c2
# ╟─da4739f0-4857-4b12-a5a9-bab91c543694
# ╟─d6f176df-b323-4d74-b609-09a50b82a15d
# ╟─4f0f5767-5c26-4ae4-a28c-20ca9ed986ee
# ╟─96c15de9-3100-4d4d-b2e0-1380cf0ec31f
# ╟─1c611a7b-a89c-4bd3-99cc-296a4ccf43a5
# ╠═af303018-55b9-4330-8e8d-a3e7d1d98697
# ╟─6dae0032-6a60-4421-a68e-a94c44416e6e
# ╟─6030b8c8-d10b-442b-875c-c9e8f5bd016a
# ╟─b7807c48-6abe-416e-bbd9-5922e9044f33
# ╠═4beb3247-28f0-4b30-b925-0061253ee304
# ╟─87bc30a2-7b7e-4877-8618-aac5b44f88db
# ╟─bb89a69b-35fc-4b42-a36c-9ac7640476c8
# ╠═e0e10a0c-464b-4185-bff7-d0a4e577cae1
# ╠═a23a7779-6b79-444c-a172-33afdf5fa204
# ╠═850df889-4876-4018-bf79-9de790d10091
# ╠═061edd0d-31a1-4f9a-842e-9b3c113d6e4b
# ╠═1cbc8875-8b23-4660-8eb6-543f31436c73
# ╠═986a7b15-a441-412f-88a7-25f9648cfcbc
# ╠═d60685cd-846f-4591-9908-2f2275a193d0
# ╟─08e918cd-0032-4f8e-be16-84f2b19d8291
# ╠═b22040f6-5726-4551-af1d-b65ae876e83c
# ╠═4951072e-5546-4825-a12c-bb917df5995c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
