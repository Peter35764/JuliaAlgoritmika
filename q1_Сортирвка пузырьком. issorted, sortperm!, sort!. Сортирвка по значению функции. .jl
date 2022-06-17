# 1)	Сортировка пузырьком. Issorted, sortperm!, sort!. Сортировка по значению функции .
# n = parse(Int, readline())
function bubblesort!(massive::AbstractArray)
    n = length(massive)
    for i in 1:n-1
        istranspose = false
        for i in firstindex(massive):lastindex(massive) - i
            if massive[i] > massive[i + 1]
                massive[i], massive[i + 1] = massive[i + 1], massive[i]
                istranspose = true
            end
        end
        if istranspose == false
            break
        end
    end
    return massive
end

a = [4, 5, 2, 6]
bubblesort!(a)

function iissorted(massive::AbstractArray)
    for i in firstindex(massive):lastindex(massive) - 1
        if massive[i] > massive[i+1]
            return false
        end
    end
    return true
end

# тут еще баблсорт, который возвращает перестановку, whatever

function bubblesortperm!(a)
n = length(a)
indexes = collect(1:n)
for k in 1:n-1
is_sorted = true
for i in 1:n-k
if a[i] > a[i+1]
a[i], a[i+1] = a[i+1], a[i]
indexes[i], indexes[i+1] = indexes[i+1], indexes[i]
is_sorted = false
end
end
if is_sorted
break
end
end
return indexes
end
bubblesortperm(a) = bubblesortperm!(deepcopy(a))
